// sm89 GEMM pipeline for the KVM megakernel.
//
// Replaces the Blackwell matmul_pipeline.cuh which uses:
//   - TMA loads (tma::load_async with barrier/semaphore)
//   - WGMMA / tensor allocator (kittens::mm / kittens::mma, s.tensor_alloc)
//
// sm89 implementation uses:
//   - cp.async group loads (group::load_async + cp_async_wait_all)
//   - warp-scope mma.sync via warp::mma on rt_fl<16, OUT_BLOCK> accumulators
//   - plain shared→global stores
//
// Output tile partitioning (16 consumer warps):
//   Each consumer warp i computes rows [i*16, (i+1)*16) × [0, OUT_BLOCK) of the output.
//   → each warp holds rt_fl<16, OUT_BLOCK> in registers.
//   For OUT_BLOCK=64: 16×64 floats/warp = 32 floats/thread → well within register budget.
//
// Shmem per stage (2-stage double-buffer):
//   a_smem = st_bf<BATCH_BLOCK, K_DIM> = 256×32×2 = 16 KB = 2 pages
//   b_smem = st_bf<OUT_BLOCK,  K_DIM> =  64×32×2 =  4 KB = 1 page (padded to 8KB)
//   per stage = 3 pages; 2 stages = 6 pages out of 10 available.

#pragma once

#include "llama_sm89.cuh"

namespace kittens::prototype::vm {

// K-tile dimension for sm89 GEMM pipeline (32 < Blackwell's 64, fits in shmem).
static constexpr int SM89_PIPELINE_K_DIM = 32;

// Number of K iterations for a given hidden dimension.
// (hidden_dim / SM89_PIPELINE_K_DIM)

template <typename Config, typename Globals,
          typename parsed_instruction,
          typename gmem_waiter,
          auto A_Ptr,   // pointer-to-member: activations (A matrix, batch × hidden)
          auto B_Ptr,   // pointer-to-member: weights    (B matrix, stored as [out, K])
          int  _num_iters>   // = in_dim / SM89_PIPELINE_K_DIM
struct matmul_pipeline_sm89 {
    static_assert(Config::NUM_CONSUMER_WARPS == 16, "sm89 pipeline requires 16 consumer warps");

    static constexpr int BATCH_BLOCK = Globals::matmul_batch_block_size; // 256
    static constexpr int OUT_BLOCK   = Globals::matmul_out_block_size;   //  64
    static constexpr int K_DIM       = SM89_PIPELINE_K_DIM;              //  32

    // Shared memory tiles for one pipeline stage.
    // A tile: [BATCH_BLOCK, K_DIM] bf16  = 256×32×2 = 16 KB = 2 pages
    // B tile: [OUT_BLOCK,   K_DIM] bf16  =  64×32×2 =  4 KB < 1 page (padded to 8KB)
    using a_st = st_bf<BATCH_BLOCK, K_DIM>;
    using b_st = st_bf<OUT_BLOCK,   K_DIM>;

    // Per-warp accumulator: [16 rows, OUT_BLOCK cols] fp32.
    // 16×64 = 1024 floats/warp = 32 floats/thread → fine.
    using acc_rt = rt_fl<16, OUT_BLOCK>;

    static constexpr int INPUT_PIPELINE_STAGES = 2;
    // Pages per stage: ceil(16KB/8KB) + ceil(4KB/8KB) = 2 + 1 = 3
    static constexpr int A_PAGES_PER_STAGE = sizeof(a_st) / Config::PAGE_SIZE
                                           + ((sizeof(a_st) % Config::PAGE_SIZE) ? 1 : 0);
    static constexpr int B_PAGES_PER_STAGE = 1;   // b_st fits in one 8KB page
    static constexpr int PAGES_PER_STAGE   = A_PAGES_PER_STAGE + B_PAGES_PER_STAGE;

    static constexpr int SEM_COUNT = 2 * INPUT_PIPELINE_STAGES + 1;

    // ── semaphore accessors ──────────────────────────────────────────────────
    __device__ static inline semaphore &inputs_arrived(state<Config> &s, int stage)  { return s.semaphores()[stage]; }
    __device__ static inline semaphore &inputs_finished(state<Config> &s, int stage) { return s.semaphores()[INPUT_PIPELINE_STAGES + stage]; }
    __device__ static inline semaphore &outputs_arrived(state<Config> &s)            { return s.semaphores()[2 * INPUT_PIPELINE_STAGES]; }

    // ── page accessors ───────────────────────────────────────────────────────
    __device__ static inline int a_page_base(state<Config> &s, int stage) {
        return s.pid(stage * PAGES_PER_STAGE);
    }
    __device__ static inline int b_page(state<Config> &s, int stage) {
        return s.pid(stage * PAGES_PER_STAGE + A_PAGES_PER_STAGE);
    }

    __device__ static inline a_st &get_a_smem(state<Config> &s, int stage) {
        // A tile spans A_PAGES_PER_STAGE consecutive pages; they are contiguous.
        return *reinterpret_cast<a_st *>(s.pages[a_page_base(s, stage)].data);
    }
    __device__ static inline b_st &get_b_smem(state<Config> &s, int stage) {
        return *reinterpret_cast<b_st *>(s.pages[b_page(s, stage)].data);
    }

    // ── semaphore init ───────────────────────────────────────────────────────
    __device__ static inline int init_semaphores(state<Config> &s) {
        for (int i = 0; i < INPUT_PIPELINE_STAGES; i++) {
            init_semaphore(inputs_arrived(s, i),  1);
            init_semaphore(inputs_finished(s, i), 2);
        }
        init_semaphore(outputs_arrived(s), 1);
        return SEM_COUNT;
    }

    // ── page-release order (same logic as original matmul_pipeline) ──────────
    __device__ static inline int release_lid(const Globals &g,
                                             typename Config::instruction_t &instruction,
                                             int &query) {
        // With 2 stages and PAGES_PER_STAGE=3, we manage 6 page slots.
        // Use the same even/odd ordering the original uses.
        auto remainder = _num_iters % INPUT_PIPELINE_STAGES;
        if (remainder == 0) {
            int ret[6] = {0,1,2,3,4,5};
            return ret[query % 6];
        } else {
            int ret[6] = {4,5,0,1,2,3};
            return ret[query % 6];
        }
    }

    // ── loader: issues cp.async loads for A and B tiles ──────────────────────
    template <int _stages_to_not_release = 0>
    __device__ static inline void loader_loop(state<Config> &s, const Globals &g,
                                              int layer_idx = 0) {
        parsed_instruction inst{s};

        if (laneid() == 0) {
            int stage = 0;

            for (int iter = 0; iter < _num_iters; iter++) {
                // Wait until consumers are done with this stage's pages.
                wait(inputs_finished(s, stage),
                     (iter % (2 * INPUT_PIPELINE_STAGES)) < INPUT_PIPELINE_STAGES);

                if (iter < INPUT_PIPELINE_STAGES) {
                    s.wait_page_ready(a_page_base(s, stage));
                    s.wait_page_ready(b_page(s, stage));
                }

                gmem_waiter::gmem_wait(g, s, inst);

                a_st &a_smem = get_a_smem(s, stage);
                b_st &b_smem = get_b_smem(s, stage);

                // Load A tile: rows [inst.row*BATCH_BLOCK .. ], K-chunk [iter]
                warp::load_async(a_smem, g.*A_Ptr, {inst.row, iter});
                // Load B tile: rows [inst.col*OUT_BLOCK .. ], K-chunk [iter, layer_idx]
                warp::load_async(b_smem, g.*B_Ptr, {layer_idx, inst.col, iter});

                // Commit + wait so consumers see coherent shmem.
                asm volatile("cp.async.commit_group;\n" ::: "memory");
                asm volatile("cp.async.wait_all;\n"     ::: "memory");

                // Signal consumers: data is ready.
                arrive(inputs_arrived(s, stage));

                stage = (stage + 1) % INPUT_PIPELINE_STAGES;
            }

            // Drain remaining pipeline stages.
            for (int i = 0; i < INPUT_PIPELINE_STAGES; i++) {
                wait(inputs_finished(s, stage),
                     ((_num_iters + i) % (2 * INPUT_PIPELINE_STAGES)) < INPUT_PIPELINE_STAGES);

                if (i == INPUT_PIPELINE_STAGES - 1)
                    arrive(outputs_arrived(s));  // all matmuls complete

                // Release pages (unless told not to for the last few stages).
                if (i < INPUT_PIPELINE_STAGES - _stages_to_not_release) {
                    // Release all pages for this stage.
                    for (int p = 0; p < PAGES_PER_STAGE; p++)
                        s.finish_page(s.pid(stage * PAGES_PER_STAGE + p), Config::NUM_CONSUMER_WARPS);
                }

                stage = (stage + 1) % INPUT_PIPELINE_STAGES;
            }
        }
    }

    // ── launcher: no-op on sm89 (consumers do warp::mma directly) ────────────
    __device__ static inline void launcher_loop(state<Config> &s, const Globals &g) {
        // On sm89 there is no separate MMA-dispatch role.
        // Signal consumers immediately so they can start accumulating.
        // (The actual mma.sync calls are in the consumer.)
#ifdef KITTENS_BLACKWELL
        // Blackwell path (should not be reached when compiling for sm89, but kept for safety).
        if (warp::laneid() == 0) {
            s.wait_tensor_ready();
            arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
        }
#endif
        // Nothing to do on sm89: consumer warps observe inputs_arrived and
        // execute warp::mma themselves inside consumer_loop.
    }

    // ── consumer compute kernel (called from the consumer struct) ─────────────
    // Each of the 16 consumer warps:
    //   - Accumulates rows [warpid()*16 .. (warpid()+1)*16) × [0 .. OUT_BLOCK) into acc_rt.
    //   - Iterates over K dimension in chunks of K_DIM.
    //   - On completion, stores the fp32 accumulator to a temporary bf16 smem tile
    //     and signals the storer.
    __device__ static inline void consumer_loop(state<Config> &s, const Globals &g,
                                                acc_rt &acc) {
        int stage = 0;
        const int my_row_start = warpid() * 16;  // this warp's 16 rows in the A tile

        for (int iter = 0; iter < _num_iters; iter++) {
            // Wait for loader to fill this stage.
            wait(inputs_arrived(s, stage),
                 (iter % (2 * INPUT_PIPELINE_STAGES)) >= INPUT_PIPELINE_STAGES);

            a_st &a_smem = get_a_smem(s, stage);
            b_st &b_smem = get_b_smem(s, stage);

            // Extract this warp's 16-row slice of A from shmem.
            // rt_bf<16, K_DIM>: 16 rows × K_DIM cols bf16 register tile.
            rt_bf<16, K_DIM> a_reg;
            rt_bf<K_DIM, OUT_BLOCK, kittens::ducks::rt_layout::col> b_reg;

            // Load a_reg from a_smem rows [my_row_start .. my_row_start+16)
            // TK's warp::load(dst, src, {row_offset, col_offset}) with coord.
            warp::load(a_reg, a_smem, {my_row_start / 16, 0});

            // Load b_reg from b_smem (full OUT_BLOCK × K_DIM tile, transposed)
            warp::load(b_reg, b_smem, {0, 0});

            // Accumulate: acc += a_reg × b_reg^T
            // warp::mma<N, T>(d, a, b, c) computes d = a × b^T + c
            if (iter == 0)
                warp::mm <transpose::N, transpose::T>(acc, a_reg, b_reg);
            else
                warp::mma<transpose::N, transpose::T>(acc, a_reg, b_reg, acc);

            // Signal loader that pages are done.
            warp::arrive(inputs_finished(s, stage));

            stage = (stage + 1) % INPUT_PIPELINE_STAGES;
        }
    }
};

} // namespace kittens::prototype::vm
