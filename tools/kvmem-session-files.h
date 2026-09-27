#pragma once

#include "kvmem/snapshot.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>

using kvmem_session_corrupt = kvmem::SnapshotCorrupt;

// Private to one server run. Only tracked files are removed; other runs and
// user files in the configured directory are never swept. Temporary files are
// charged before the first write and remain charged if removal fails.
class kvmem_session_files {
public:
    kvmem_session_files(const std::filesystem::path & root, uint64_t limit) : limit_(limit) {
        std::filesystem::create_directories(root);
        for (int attempt = 0; attempt < 32; ++attempt) {
            dir_ = root / ("run-" + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) +
                           "-" + std::to_string(std::random_device{}()));
            if (std::filesystem::create_directory(dir_)) {
                std::filesystem::permissions(dir_, std::filesystem::perms::owner_all,
                                             std::filesystem::perm_options::replace);
                return;
            }
        }
        throw std::runtime_error("cannot create private session cache directory");
    }
    ~kvmem_session_files() {
        for (const auto & entry : files_) { std::error_code ec; std::filesystem::remove(entry.second.path, ec); }
        std::error_code ec; std::filesystem::remove(dir_, ec);
    }
    kvmem_session_files(const kvmem_session_files &) = delete;
    kvmem_session_files & operator=(const kvmem_session_files &) = delete;
    uint64_t bytes() const { return bytes_; }
    uint64_t limit() const { return limit_; }
    uint64_t available() const { return std::filesystem::space(dir_).available; }
    const std::filesystem::path & directory() const { return dir_; }
    // Optional deterministic fault injection for the portable transfer tests.
    std::function<void(const char *, int, uint32_t)> fault;
    std::filesystem::path path(int id, uint32_t chunk = 0) const { return files_.at({id, chunk}).path; }
    bool contains(int id) const {
        auto it = files_.lower_bound({id, 0}); return it != files_.end() && it->first.first == id;
    }
    bool contains_chunk(int id, uint32_t chunk) const { return files_.count({id, chunk}) != 0; }
    bool ready(int id, uint32_t chunk) const {
        auto it = files_.find({id, chunk}); return it != files_.end() && it->second.ready;
    }
    uint64_t chunk_bytes(int id, uint32_t chunk) const {
        auto it = files_.find({id, chunk}); return it == files_.end() ? 0 : it->second.bytes;
    }
    uint64_t session_bytes(int id) const {
        uint64_t n = 0;
        for (auto it = files_.lower_bound({id, 0}); it != files_.end() && it->first.first == id; ++it) n += it->second.bytes;
        return n;
    }
    bool erase(int id) {
        bool ok = true;
        for (auto it = files_.lower_bound({id, 0}); it != files_.end() && it->first.first == id;) {
            const auto chunk = it++->first.second;
            if (!erase_chunk(id, chunk)) ok = false;
        }
        return ok;
    }
    bool erase_chunk(int id, uint32_t chunk) {
        const auto it = files_.find({id, chunk});
        if (it == files_.end()) return true;
        try { if (fault) fault("erase", id, chunk); } catch (...) { return false; }
        std::error_code ec;
        std::filesystem::remove(it->second.path, ec);
        if (ec) return false;
        bytes_ -= it->second.bytes; files_.erase(it); return true;
    }
    template<class Write> void save(int id, uint64_t payload_bytes, const Write & write) {
        save_chunk(id, 0, payload_bytes, write);
    }
    template<class Write> void save_chunk(int id, uint32_t chunk, uint64_t payload_bytes, const Write & write) {
        if (contains_chunk(id, chunk) || payload_bytes > UINT64_MAX - 8 || payload_bytes + 8 > limit_ - bytes_)
            throw std::runtime_error("session disk quota exceeded");
        const auto total = payload_bytes + 8;
        if (available() < total)
            throw std::runtime_error("insufficient free space for session snapshot");
        const auto stem = std::to_string(id) + "-" + std::to_string(chunk);
        const auto temp = dir_ / (stem + ".tmp");
        auto final = dir_ / (stem + ".kv");
        files_.emplace(std::make_pair(id, chunk), record{temp, total, false}); bytes_ += total;
        try {
            if (fault) fault("write", id, chunk);
            std::ofstream file(temp, std::ios::binary | std::ios::trunc);
            file.exceptions(std::ios::badbit | std::ios::failbit);
            kvmem::SnapshotWriter out([&](const void * p, size_t n) {
                if (fault) fault("write_data", id, chunk);
                file.write((const char *)p, std::streamsize(n));
            });
            write(out);
            if (out.bytes() != payload_bytes) throw std::runtime_error("session changed during snapshot");
            const uint64_t hash = out.hash(); file.write((const char *)&hash, sizeof(hash));
            file.flush(); file.close();
            if (fault) fault("rename", id, chunk);
            std::filesystem::rename(temp, final);
            auto & record = files_.at({id, chunk});
            record.path.swap(final); // noexcept: never orphan a renamed file
            record.ready = true;
        } catch (...) { erase_chunk(id, chunk); throw; }
    }
    template<class Read> void load(int id, const Read & read) const {
        load_chunk(id, 0, read);
    }
    template<class Read> void load_chunk(int id, uint32_t chunk, const Read & read) const {
        const auto & record = files_.at({id, chunk});
        if (fault) fault("read", id, chunk);
        std::error_code ec;
        const auto size = std::filesystem::file_size(record.path, ec);
        if (ec == std::errc::no_such_file_or_directory) throw kvmem_session_corrupt("missing session snapshot");
        if (ec) throw std::filesystem::filesystem_error("cannot inspect session snapshot", record.path, ec);
        if (!record.ready || size != record.bytes || record.bytes < 8)
            throw kvmem_session_corrupt("truncated session snapshot");
        std::ifstream file(record.path, std::ios::binary);
        file.exceptions(std::ios::badbit | std::ios::failbit);
        kvmem::SnapshotReader in([&](void * p, size_t n) {
            if (fault) fault("read_data", id, chunk);
            file.read((char *)p, std::streamsize(n));
        }, record.bytes - 8);
        read(in);
        uint64_t hash = 0; file.read((char *)&hash, sizeof(hash));
        if (in.remaining() || hash != in.hash()) throw kvmem_session_corrupt("session snapshot checksum mismatch");
        if (fault) fault("validated", id, chunk);
    }
private:
    struct record { std::filesystem::path path; uint64_t bytes; bool ready; };
    std::filesystem::path dir_;
    std::map<std::pair<int, uint32_t>, record> files_;
    uint64_t bytes_ = 0, limit_;
};
