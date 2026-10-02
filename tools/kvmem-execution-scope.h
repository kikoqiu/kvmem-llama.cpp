#pragma once
#include "llama-kvmem-hooks.h"

class kvmem_execution_scope {
public:
    explicit kvmem_execution_scope(llama_kvmem_execution_state * state)
        : previous_(llama_kvmem_execution_exchange(state)) {}
    ~kvmem_execution_scope() { llama_kvmem_execution_exchange(previous_); }
    kvmem_execution_scope(const kvmem_execution_scope &) = delete;
    kvmem_execution_scope & operator=(const kvmem_execution_scope &) = delete;
private:
    llama_kvmem_execution_state * previous_;
};
