# Architecture (P0)

KVMem is a block-sparse KV working-set manager. llama.cpp will own inference;
this library owns selection, tiering, and (later) window assembly.

```
kvmem/          host policy + CPU/NVMe  (this repo, no llama.cpp headers)
src/adapter/    llama_memory_i wrapper  (P1+)
llama.cpp/      llama.cpp checkout + thin patches (P1+)
```

GPU attention cache is a **bounded block-slot pool** of size
`budget + gen_reserve`. Product default GPU KV type is llama.cpp **q8_0**
for K and V (`--kv-dtype q8_0`; `f16` or `q4_0` to override). Cell `pos` is
the original monotonic token position (slot index is not a RoPE coordinate).
Restore is packed GPU-format memcpy at that orig pos — no unrotated raw-K
and no re-RoPE on the product path. Retrieval scores **mean-K** (F32,
pre-RoPE, captured at first write). Packed K/V for a full block are
copied to host asynchronously when the block fills, overlapping later
prefill; eviction then skips if the copy exists. After retrieval pin,
decode keeps a GPU running sum of pre-RoPE K and writes mean-K when a
block fills (accepted tokens only; MTP drafts are not counted). Cold
stage-in is `copy_k_gpu` / `copy_v_gpu` + slab H2D. `llama-kvmem-server`
reuses the GPU prefix across requests (token LCP); the retrieval query
is the last `role=user` span. Chat tools reuse llama.cpp `common/chat`
+ `common_sampler`; the server does not execute tools. Packed transfers use a **32 MiB**
GPU slab. GPU-format CPU/NVMe scratch is allocated only when those tiers
are on. Each logical block occupies one slot of `block_tokens` cells.
Reselect is a `KvMemPlan` diff: resident selected blocks stay in their
slot; only `stage_out` cells are `seq_rm`'d. Flash Attention is not
modified. P1 recency does not re-RoPE and does not resurrect dropped
blocks.

With the default `--parallel 1`, the host store is per conversation; the GPU
working set and the `llama_context` stay single. `--kvmem-conversations N` keeps N host stores alive and
time-multiplexes them, so a conversation that returns after another was served
does not have to be reprocessed. A switch drains the whole working set to host
and rebuilds the incoming store's through the drain-and-restage path that
already runs inside a conversation (the host fallback in
`layout_gpu_slots_by_orig_pos` and `write_block_to_gpu` in
`src/adapter/llama-memory-kvmem.cpp`), not a second implementation. Requests
stay serialized and `n_seq_max` stays 1; the recurrent half is still a
server-side byte snapshot restored per request.

### Multiple lanes and conversations

`--kvmem --parallel P --kvmem-conversations N`, with P=2..4, creates P independent target
contexts sharing immutable model weights, with one global host conversation
pool. Effective N is at least P. Each lane owns its GPU window, recurrent state,
MTP follower (when enabled), capture state and transfer scratch. GPU context,
retrieval and generation budgets apply per lane; `--kvmem-cpu-gb` applies per
host store. Startup verifies compatible KV geometry across lanes.

Admission uses FIFO among ready, eligible conversation heads. Same-ID requests
serialize; unrelated ready requests can use any free lane. A warm residency is
preferred, but is never a permanent assignment. A switch quiesces the outgoing
lane, invalidates its attention/MTP cells and transfers a backend-neutral host
bundle by ownership, then restores incoming checkpoints. GPU buffers are reused.

One shared projector can run on CPU or another device, including the lane GPU.
Stateful tokenization/encoding is serialized before lane admission; immutable
prepared embeddings are decoded independently. The embedding cache and in-flight
references share a 128 MiB bound. Text and image requests support optional MTP on
one CUDA target GPU with at least `2 * P` HTTP workers. Video/audio, multiple target
GPUs and NVMe/session disk are outside multi-lane support. Throughput depends on the
workload and device. See [the complete design](multi-lane-conversations.md).

Hardware split on this machine: RTX 5050 (GPU 0) for models < 27B;
RTX 5090 (GPU 1) for 27B. Details in `scripts/gpu.sh` and
`docs/modification-plan.md`.

## Prefill pressure policy

A prefill longer than the pool evicts while it is still running. The recency
policy keeps the sink prefix, then fills the prefill budget with the newest
blocks. Decode-time retrieval then re-selects with the request query, so the two
stages can disagree and a re-prefilled context lands on a different resident set
than the same context served from a live window.

`--kvmem-prefill-method retrieval` (the server default) makes the prefill
pressure use the same scored selection as decode: sink, incoming rows, the
`--kvmem-recent-tokens` suffix pin, then the best-scoring blocks. The request
query is normally still ahead of the prefill cursor, so the newest
user-message span that has already been prefilled stands in for it.
`llama-kvmem-server` passes every user span of the rendered prompt
(`--kvmem-prefill-query-max-tokens`, default 128, keeps only each span tail);
the adapter scores blocks against the mean-Q of that span and reselects inside
the prefill budget, keeping the request query span mandatory. No user span yet
(system prompt, first message) means no query, and the policy stays recency.
Raw completion and the CLI have no chat roles, so they keep recency.

## Generation length vs `gen_reserve`

GPU pool = `budget` (selected working set) + `gen_reserve` (decode slack).
After retrieval the selected blocks are **pinned**: decode must not
recency-reselect and drop resurrected blocks. New tokens only take free
slots in `gen_reserve`. `--kvmem-gen-exceed` (server and CLI, default
`retrieval`) decides what happens when those slots fill.

`retrieval`: the last GPU block is full and no slot is free, so the adapter
reselects across the whole pool once - `budget` plus `gen_reserve`.
Candidates are every stored block, including the rows decode just wrote, and
they are ranked by the same retrieval score as any other history block. Only
the losers are staged out. Mandatory: the request query, the incoming rows,
and the block holding the row before the ubatch, which the next token
attends. Outgoing blocks are harvested to the packed host copy before they
leave, so a later stage-in restores them. Spilled generation blocks carry
decode mean-K and are selectable again by a later turn's query.

`error`: the v1 behavior. `prepare_working_set` fails with
`no free GPU slot for block N` (`llama_decode(gen) failed rc=1`), and one
generation cannot exceed `--kvmem-gen-reserve` (16384 on IQ3, 12288 on IQ4,
CLI default 256), thinking included. `llama-kvmem-server` caps `max_tokens`
at the reserve in this mode, and the reserve is also the output length a request
gets when it omits `max_tokens`. With the retrieval default the logical context
(`-c`) is the only limit, and an omitted `max_tokens` uses the whole pool
(`budget + gen_reserve`).

The swap is an ordinary retrieval reselect, so it needs the packed host copy
of every block it drops. Packed V exists only with flash attention
(`-fa on`, which the 16 GiB recipes use); a transposed-V build falls back to
the error path.

### Why not steal slots from the selected set

The pool is already two partitions. Selected KV occupies `budget` slots
and stays pinned. Generation occupies `gen_reserve` slots. When those
are full, the card is not “choose generation or retrieval”: only the
reserve partition is full. Unpinning and recency-evicting the selected
blocks would drop the query’s retrieved facts, so the swap reselects with
the query still mandatory. Growing `budget +
gen_reserve` on 16 GiB is also out: recipes already sit near 15.5 GiB.
Streaming the whole generation through VRAM would bring back the
adaptive-KV-streaming cost curve. The optional NVMe session cache stores idle
sessions, while generation spill remains in host RAM.

### What the swap keeps

No third pool and no extra VRAM. `gen_reserve` is “how much of **this
turn’s** output attention can still see”, not a hard max length.

Rules the swap follows:

1. Mandatory (the selector keeps these first): the incoming rows, the block
   that holds the row before the ubatch, and the request query span.
   Mandatory is filled newest-first, so a query span that cannot fit loses its
   oldest blocks.
2. Then the sink prefix (`--kvmem-sink-tokens`, default one block) and the
   `--kvmem-recent-tokens` suffix keep their existing policy.
3. Every other block - this turn's already written blocks and older history
   alike - is an ordinary candidate. They compete on the same retrieval score
   (query mean-Q against block mean-K, newest block wins a tie); the lowest
   scorers are staged out. No rule evicts this turn's blocks first.
4. Every block that loses is harvested first (packed K/V, decode mean-K), so
   the host store stays the authority and a later turn can stage it back in.
5. Grain is one reselect, not one block. `--kvmem-block-tokens` (128 on the
   16 GiB recipes) sets the transfer unit; a swap moves only the blocks the
   new window does not keep.

After a swap, GPU still holds the new selected window plus the tail block
that decode is writing. MTP’s follower pool uses the same slot indices; the
same blocks leave the draft cache. GDN / recurrent state is updated every
token and does not live in these attention slots, so it is not part of the
swap.

Spilled gen KV stays in the host store. This turn’s Flash Attention does not
see it; the next turn’s retrieval can select it.

**Tail size.** Keep the current recipe reserves: 16K (IQ3) / 12K (IQ4).
That VRAM is already paid. It covers the default 4096 thinking budget plus a
normal answer without a swap. 4K would cover thinking but drop long code;
32K would steal from `budget` or blow 16 GiB.

**Quality.** The hard crash goes away. A 32K thinking dump still only
attends to the last 12K/16K of itself plus what the query scores. The
system-prompt cap remains useful. Seeing earlier thoughts in the same turn
would be a later step (retrieve from spilled gen blocks using the current
generation as query), not part of this cut.

**Recency decode.** Its `block_count() > budget` mis-trigger is the same
class of pin bug and stays out of this cut.

## P7: MTP shares the slot-pool (plan B)

Logically long context must not grow a full-length MTP KV (plan A). The
draft context (`LLAMA_CONTEXT_TYPE_MTP`) gets a **follower** slot-pool
the same size as the target attention cache, same block IDs and slot
indices, original `pos` on cells. Speculative decoding stays in
llama.cpp `draft-mtp`. Details: `docs/kvmem-mtp-plan.md`.
