#include "llama-kvmem-capture.h"
#include "llama-kvmem-execution.h"

#include "llama-kvmem-hooks.h"
#include "llama-memory-kvmem.h"
#include "llama-memory-kvmem-mtp.h"

#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>


void kvmem_capture_bind(llama_memory_kvmem * mem) {
    kvmem_current_execution().memory = mem;
}

llama_memory_kvmem * kvmem_capture_active() {
    return kvmem_current_execution().memory;
}

void kvmem_capture_unbind(llama_memory_kvmem * mem) {
    if (kvmem_current_execution().memory == mem) {
        kvmem_current_execution().memory = nullptr;
    }
}

void kvmem_mtp_bind(llama_memory_kvmem_mtp * mem) {
    kvmem_current_execution().mtp = mem;
}

void kvmem_mtp_unbind(llama_memory_kvmem_mtp * mem) {
    if (kvmem_current_execution().mtp == mem) {
        kvmem_current_execution().mtp = nullptr;
    }
}

void kvmem_capture_note_ubatch(const std::vector<llama_pos> & pos) {
    if (kvmem_current_execution().memory) {
        kvmem_current_execution().memory->note_ubatch_pos(pos);
    }
}

void kvmem_capture_reset_q() {
    if (kvmem_current_execution().memory) {
        kvmem_current_execution().memory->reset_query_acc();
    }
}

void kvmem_capture_register(struct ggml_tensor * t, int il, char which) {
    if (kvmem_current_execution().mtp && kvmem_current_execution().mtp->is_mtp_layer(il)) {
        kvmem_current_execution().mtp->register_capture(t, il, which);
        return;
    }
    if (kvmem_current_execution().memory) {
        kvmem_current_execution().memory->register_capture(t, il, which);
    }
}

void kvmem_capture_on_new_graph(int is_mtp) {
    if (is_mtp) {
        if (kvmem_current_execution().mtp) {
            kvmem_current_execution().mtp->capture_on_new_graph();
        }
        return;
    }
    if (kvmem_current_execution().memory) {
        kvmem_current_execution().memory->capture_on_new_graph();
    }
}

void kvmem_capture_harvest_ubatch(struct ggml_backend_sched * sched, int is_mtp) {
    if (is_mtp) {
        if (kvmem_current_execution().mtp) {
            kvmem_current_execution().mtp->harvest_pending(sched);
        }
        return;
    }
    if (kvmem_current_execution().memory) {
        kvmem_current_execution().memory->harvest_pending(sched);
    }
}

static bool ubatch_overlaps_query(uint32_t n_tokens, uint32_t n_pos, const llama_pos * pos) {
    if (kvmem_current_execution().memory) return kvmem_current_execution().memory->query_overlaps(n_tokens, pos);
    const llama_kvmem_params * kp = llama_kvmem_get_params();
    if (!kp || !kp->enabled || kp->method != 1 || kp->query_begin < 0 || n_tokens == 0) {
        return false;
    }
    const int32_t qb = kp->query_begin;
    const int32_t qe = kp->query_end > 0 ? kp->query_end : (1 << 30);
    (void) n_pos;
    if (!pos) {
        return true;
    }
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const llama_pos p = pos[i];
        if (p >= qb && p < qe) {
            return true;
        }
    }
    return false;
}

bool kvmem_ubatch_needs_q_capture(uint32_t n_tokens, uint32_t n_pos, const llama_pos * pos) {
    return ubatch_overlaps_query(n_tokens, n_pos, pos);
}

bool kvmem_capture_can_reuse(uint32_t n_tokens, uint32_t n_pos, const llama_pos * pos, int is_mtp) {
    if (is_mtp) {
        return true;
    }
    if (!kvmem_current_execution().memory) {
        return true;
    }
    return kvmem_current_execution().memory->capture_can_reuse(n_tokens, n_pos, pos);
}

bool llama_kvmem_eval_callback(struct ggml_tensor * /*t*/, bool /*ask*/, void * /*user_data*/) {
    // Do not observe nodes: the scheduler splits and synchronizes on every
    // true ask. Harvest is llama_kvmem_harvest_ubatch after the full graph.
    return false;
}

void llama_kvmem_register_capture(struct ggml_tensor * t, int il, char which) {
    kvmem_capture_register(t, il, which);
}

void llama_kvmem_capture_on_new_graph(int is_mtp) {
    kvmem_capture_on_new_graph(is_mtp);
}

void llama_kvmem_harvest_ubatch(struct ggml_backend_sched * sched, int is_mtp) {
    kvmem_capture_harvest_ubatch(sched, is_mtp);
}

bool llama_kvmem_ubatch_needs_q_capture(uint32_t n_tokens, uint32_t n_pos, const llama_pos * pos) {
    return kvmem_ubatch_needs_q_capture(n_tokens, n_pos, pos);
}

bool llama_kvmem_capture_can_reuse(uint32_t n_tokens, uint32_t n_pos, const llama_pos * pos, int is_mtp) {
    return kvmem_capture_can_reuse(n_tokens, n_pos, pos, is_mtp);
}

uint64_t llama_kvmem_capture_stamp(void) {
    const auto * memory = kvmem_current_execution().memory;
    return memory ? memory->capture_stamp() : 0;
}
