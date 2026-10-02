#pragma once
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// Slot admission only; inference never runs under this mutex. Queued requests
// retain FIFO order, and disconnected waiters do not consume a lane.
class kvmem_lane_pool {
public:
    class ticket {
    public:
        ~ticket() {
            if (!pool_) return;
            auto * pool = pool_; // erase_waiter clears this ticket's owner.
            std::lock_guard<std::mutex> lock(pool->mu_);
            pool->erase_waiter(this);
            pool->changed_.notify_all();
        }
        ticket(const ticket &) = delete;
        ticket & operator=(const ticket &) = delete;
    private:
        friend class kvmem_lane_pool;
        ticket(kvmem_lane_pool & pool, std::string key) : pool_(&pool), key_(std::move(key)) {}
        kvmem_lane_pool * pool_;
        std::string key_;
        bool ready_ = false;
        std::function<int(const std::vector<bool> &, bool)> select_;
    };
    class lease {
    public:
        size_t index;
        ~lease() { release(); }
        lease(const lease &) = delete;
        lease & operator=(const lease &) = delete;
        void release() {
            if (!pool_) return;
            auto * pool = pool_;
            pool_ = nullptr;
            {
                std::lock_guard<std::mutex> lock(pool->mu_);
                pool->busy_[index] = false;
                pool->keys_[index].clear();
            }
            pool->changed_.notify_all();
        }
    private:
        friend class kvmem_lane_pool;
        lease(size_t lane) : index(lane), pool_(nullptr) {}
        kvmem_lane_pool * pool_;
    };
    explicit kvmem_lane_pool(size_t count) : busy_(count, false), keys_(count) {
        if (count == 0) throw std::invalid_argument("lane pool cannot be empty");
    }

    std::shared_ptr<ticket> enqueue(std::string key = {}) {
        auto result = std::shared_ptr<ticket>(new ticket(*this, std::move(key)));
        std::lock_guard<std::mutex> lock(mu_);
        waiting_.push_back(result.get());
        changed_.notify_all();
        return result;
    }

    template<class Cancelled>
    bool wait_turn(const std::shared_ptr<ticket> & request, Cancelled cancelled) {
        std::unique_lock<std::mutex> lock(mu_);
        while (!group_ready(request.get())) {
            if (cancelled()) { erase_waiter(request.get()); return false; }
            changed_.wait_for(lock, std::chrono::milliseconds(50));
        }
        return !cancelled();
    }

    // Select is read-only until reserve=true. Admission never holds the lock
    // during media preparation, KV transfers or model execution.
    template<class Cancelled, class Select>
    std::shared_ptr<lease> acquire(const std::shared_ptr<ticket> & request,
                                    Cancelled cancelled, Select select) {
        std::unique_lock<std::mutex> lock(mu_);
        request->select_ = std::move(select);
        request->ready_ = true;
        for (;;) {
            if (cancelled()) {
                erase_waiter(request.get());
                changed_.notify_all();
                return {};
            }
            if (request->select_(busy_, false) == -2) {
                request->ready_ = false;
                changed_.notify_all();
                return {}; // Prepare missing input and resume this ticket.
            }
            ticket * first = nullptr;
            for (auto * candidate : waiting_) {
                if (candidate->ready_ && group_ready(candidate) && candidate->select_(busy_, false) >= 0) {
                    first = candidate;
                    break;
                }
            }
            if (first == request.get()) {
                // Allocate before the callback reserves a host store. Once it
                // succeeds, all remaining ownership changes are nonthrowing.
                auto result = std::shared_ptr<lease>(new lease(0));
                std::string key = request->key_;
                const int lane = request->select_(busy_, true);
                if (lane >= 0 && (size_t) lane < busy_.size() && !busy_[lane]) {
                    result->index = (size_t)lane;
                    result->pool_ = this;
                    busy_[lane] = true;
                    keys_[lane].swap(key);
                    next_ = ((size_t) lane + 1) % busy_.size();
                    erase_waiter(request.get());
                    changed_.notify_all();
                    return result;
                }
            }
            changed_.wait_for(lock, std::chrono::milliseconds(50));
        }
    }

    template<class Cancelled>
    std::shared_ptr<lease> acquire(Cancelled cancelled) {
        return acquire(enqueue(), cancelled, [&](const std::vector<bool> & busy, bool) {
            for (size_t n = 0; n < busy.size(); ++n) {
                const size_t lane = (next_ + n) % busy.size();
                if (!busy[lane]) return (int) lane;
            }
            return -1;
        });
    }
private:
    std::mutex mu_;
    std::condition_variable changed_;
    std::vector<bool> busy_;
    std::vector<std::string> keys_;
    std::deque<ticket *> waiting_;
    size_t next_ = 0;
    void erase_waiter(ticket * request) {
        auto it = std::find(waiting_.begin(), waiting_.end(), request);
        if (it != waiting_.end()) waiting_.erase(it);
        request->pool_ = nullptr;
    }
    bool group_ready(const ticket * request) const {
        if (request->key_.empty()) return true;
        if (std::find(keys_.begin(), keys_.end(), request->key_) != keys_.end()) return false;
        for (auto * earlier : waiting_) {
            if (earlier == request) return true;
            if (earlier->key_ == request->key_) return false;
        }
        return true;
    }
};
