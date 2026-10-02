#ifdef NDEBUG
#undef NDEBUG
#endif
#include "kvmem-lane-pool.h"
#include <atomic>
#include <cassert>
#include <future>
#include <iostream>

int main() {
    using namespace std::chrono_literals;
    kvmem_lane_pool pool(2);
    auto available = [] { return false; };
    {
        auto rejected_before_admission = pool.enqueue("invalid");
    } // Removing an unprepared ticket must still notify through its old owner.
    auto first = pool.acquire(available);
    auto second = pool.acquire(available);
    assert(first && second && first->index != second->index);

    std::promise<void> queued;
    std::atomic<bool> cancel{false};
    bool announced = false;
    auto waiting = std::async(std::launch::async, [&] {
        return pool.acquire([&] {
            if (!announced) { announced = true; queued.set_value(); }
            return cancel.load();
        });
    });
    queued.get_future().wait();
    assert(waiting.wait_for(100ms) == std::future_status::timeout);
    cancel = true;
    assert(waiting.wait_for(2s) == std::future_status::ready);
    assert(!waiting.get());

    // Two enqueued waiters are admitted in order, with no duplicate lease.
    std::promise<void> entered_a, entered_b;
    auto wait_on = [&](std::promise<void>& entered) {
        return std::async(std::launch::async, [&pool, &entered] {
            bool announced = false;
            return pool.acquire([&] {
                if (!announced) { announced = true; entered.set_value(); }
                return false;
            });
        });
    };
    auto a = wait_on(entered_a);
    entered_a.get_future().wait();
    auto b = wait_on(entered_b);
    entered_b.get_future().wait();
    const auto freed = first->index;
    first.reset();
    assert(a.wait_for(2s) == std::future_status::ready);
    auto third = a.get();
    assert(third->index == freed);
    assert(b.wait_for(100ms) == std::future_status::timeout);
    second.reset();
    assert(b.wait_for(2s) == std::future_status::ready);
    auto fourth = b.get();
    assert(fourth->index != third->index);
    third->release();
    third->release(); // explicit completion plus destruction is idempotent
    auto reused = pool.acquire(available);
    assert(reused->index == freed);
    for (size_t count : {2, 4}) {
        kvmem_lane_pool sessions(count);
        auto choose = [](const std::vector<bool> & busy, bool) {
            for (size_t n = 0; n < busy.size(); ++n) if (!busy[n]) return (int)n;
            return -1;
        };
        auto current = sessions.acquire(sessions.enqueue("A"), available, choose);
        auto followup = sessions.enqueue("A");
        auto next_turn = std::async(std::launch::async, [&] {
            assert(sessions.wait_turn(followup, available));
            return sessions.acquire(followup, available, choose);
        });
        assert(next_turn.wait_for(100ms) == std::future_status::timeout);
        auto preparing = sessions.enqueue("C");
        auto unrelated = sessions.acquire(sessions.enqueue("B"), available, choose);
        assert(unrelated && unrelated->index != current->index);
        unrelated.reset();
        current.reset();
        assert(next_turn.wait_for(2s) == std::future_status::ready);
        auto resumed = next_turn.get();
        assert(resumed);
        resumed.reset();
        assert(!sessions.acquire(preparing, available, [](const std::vector<bool> &, bool) { return -2; }));
        auto ready = sessions.acquire(preparing, available, choose);
        assert(ready); // The original queue position survives extra preparation.
    }
    {
        kvmem_lane_pool wider(4);
        std::vector<std::shared_ptr<kvmem_lane_pool::lease>> occupied;
        for (size_t lane = 0; lane < 4; ++lane) {
            auto held = wider.acquire(available);
            assert(held && held->index == lane);
            occupied.push_back(std::move(held));
        }
        std::promise<void> queued_four;
        bool reported = false;
        auto fifth = std::async(std::launch::async, [&] {
            return wider.acquire([&] {
                if (!reported) { reported = true; queued_four.set_value(); }
                return false;
            });
        });
        queued_four.get_future().wait();
        assert(fifth.wait_for(100ms) == std::future_status::timeout);
        occupied[3].reset();
        assert(fifth.wait_for(2s) == std::future_status::ready);
        assert(fifth.get()->index == 3); // Only the fourth lane became free.
    }
    std::cout << "lane pool: passed\n";
}
