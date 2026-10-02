#pragma once

#include "llama-kvmem-hooks.h"
#include <memory>

class llama_memory_kvmem;
class llama_memory_kvmem_mtp;
struct kvmem_conv_pool;
struct kvmem_gpu_state;

// One target context and its MTP follower form one execution domain. HTTP
// workers may change between turns; mutable state belongs to the domain,
// never to the worker thread. The thread-local pointer only selects it.
struct llama_kvmem_execution_state {
    llama_kvmem_execution_state();
    ~llama_kvmem_execution_state();
    llama_kvmem_execution_state(const llama_kvmem_execution_state &) = delete;
    llama_kvmem_execution_state & operator=(const llama_kvmem_execution_state &) = delete;
    llama_kvmem_params params{};
    llama_memory_kvmem * memory = nullptr;
    llama_memory_kvmem_mtp * mtp = nullptr;
    std::unique_ptr<kvmem_conv_pool> conversations;
    kvmem_gpu_state * gpu = nullptr;
};

llama_kvmem_execution_state & kvmem_current_execution();
