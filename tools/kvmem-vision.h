#pragma once

#include "llama.h"
#include "mtmd.h"

#include <functional>
#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct server_tokens;
struct common_chat_msg_delimiters;
struct common_chat_msg_spans;

// Native media chunks and prefix matching, with a row view for the existing sampler.
class kvmem_prompt {
public:
    explicit kvmem_prompt(const std::vector<llama_token> & tokens);
    explicit kvmem_prompt(std::shared_ptr<server_tokens> native);
    std::vector<llama_token> tokens;
    bool has_media() const;
    size_t common_prefix(const kvmem_prompt & other) const;
    llama_pos model_pos(size_t row) const;
    size_t media_end(size_t row) const;
    const mtmd_input_chunk * chunk(size_t row) const;
    std::vector<std::pair<uint32_t, uint32_t>> media_ranges() const;
    common_chat_msg_spans message_spans(const common_chat_msg_delimiters & delimiters) const;
    std::vector<std::pair<uint32_t, std::string>> media_identity() const;
    std::shared_ptr<kvmem_prompt> with_generated(const std::vector<llama_token> & gen) const;
    std::shared_ptr<kvmem_prompt> prefix(size_t rows) const;
    // Matching/position metadata without the original image/audio tensors.
    std::shared_ptr<kvmem_prompt> cache_index() const;
    size_t index_bytes() const;
    bool media_ready(size_t begin) const;
    void release_media() { embeddings_.clear(); }
    uint32_t encode_calls = 0;
    double encode_ms = 0;
private:
    friend class kvmem_vision;
    std::map<size_t, std::shared_ptr<const std::vector<float>>> embeddings_;
    std::shared_ptr<server_tokens> native_;
    std::map<size_t, llama_pos> position_offsets_ {{0, 0}};
};

class kvmem_vision {
public:
    kvmem_vision(llama_model * model, const std::string & path, bool gpu,
                 ggml_backend_dev_t device, int min_tokens, int max_tokens, int n_threads,
                 float video_fps);
    ~kvmem_vision();
    bool supports_video() const;
    std::shared_ptr<kvmem_prompt> tokenize(const std::string & prompt, const std::vector<std::vector<uint8_t>> & files);
    void prepare(kvmem_prompt & prompt, size_t begin = 0,
                 const std::function<bool()> & cancelled = {});
    int decode(llama_context * ctx, const kvmem_prompt & prompt, size_t row, int n_batch,
               const std::function<int(llama_batch)> & dispatch);
    size_t cache_bytes() const { return cache_bytes_.load(); }
private:
    struct accounting {
        std::atomic<size_t> bytes {0};
        std::condition_variable changed;
    };
    std::shared_ptr<accounting> accounting_ = std::make_shared<accounting>();
    std::mutex mu_;
    std::timed_mutex preparation_mu_;
    mtmd_context * ctx_ = nullptr;
    int n_embd_ = 0;
    float video_fps_ = 2.0f;
    std::atomic<size_t> cache_bytes_ {0};
    struct entry { std::shared_ptr<const std::vector<float>> embd; uint64_t used = 0; };
    std::map<std::string, entry> cache_;
    uint64_t clock_ = 0;
};

// Rejects media that the configured projector/build does not support.
std::string kvmem_parse_media_messages(const std::string & body, bool allow_images, bool allow_video,
                                      std::vector<std::vector<uint8_t>> & files);
