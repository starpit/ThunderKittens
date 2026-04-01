// sm89 port of attention_decode.cu
// The consumer (Q@K^T, softmax, @V) already uses warp-scope mma.sync — no WGMMA!
// Changes vs Hopper:
//   - Loader: tma::expect/load_async → cp.async (group/warp::load_async) + arrive
//   - Launcher: s.wait_tensor_ready() → no-op (sm89 has no tensor allocator)
//   - Storer: tma::store_async → warp::store
//   - NUM_STAGES computed from shmem budget (sm89 has fewer pages)
//   - kv_cache_t has no tma::descriptor on sm89

#include "llama_sm89.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm {

using globals = llama_sm89_globals;
using config  = llama_sm89_config;

template <typename Config = config, typename Globals = globals>
struct attention_decode {
    static constexpr int opcode             = OPCODE_GQA_AttentionDecode;
    static constexpr int GQA_RATIO          = Globals::num_attention_heads / Globals::num_kv_heads;
    static constexpr int ATTN_BATCH_BLOCK_SIZE = 4;

    static_assert(GQA_RATIO == 4, "GQA_RATIO must be 4 (for LLaMA-style GQA).");

    static constexpr int head_dim       = Globals::head_dim;
    static constexpr int kv_block_size  = Globals::kv_block_size;

    // KV tile: st_bf<kv_block_size, head_dim> per batch item
    using kv_st    = st_bf<kv_block_size, head_dim>;
    // Pages needed per stage: ceil(ATTN_BATCH_BLOCK_SIZE * kv_st / PAGE_SIZE) for K + same for V
    static constexpr int KV_TILE_BYTES  = sizeof(kv_st) * ATTN_BATCH_BLOCK_SIZE;
    static constexpr int K_PAGES        = (KV_TILE_BYTES + Config::PAGE_SIZE - 1) / Config::PAGE_SIZE;
    static constexpr int V_PAGES        = K_PAGES;
    static constexpr int PAGES_PER_STAGE = K_PAGES + V_PAGES;
    // Use as many stages as we can fit in the page budget.
    static constexpr int NUM_STAGES     = Config::NUM_PAGES / PAGES_PER_STAGE;
    static_assert(NUM_STAGES >= 1, "Not enough shmem pages for even 1 attention stage.");

    using q_rt       = rt_bf<16, head_dim>;
    using q_st       = st_bf<16, head_dim>;
    using k_rt       = rt_bf<kv_block_size, head_dim>;
    using v_rt       = rt_bf<kv_block_size, head_dim, col_l>;
    using attn_fl_rt = rt_fl<16, kv_block_size>;
    using attn_bf_rt = rt_bf<16, kv_block_size>;
    using max_vec_rv = col_vec<rt_fl<16, head_dim>>;
    using norm_vec_rv= col_vec<rt_fl<16, head_dim>>;
    using o_rt       = rt_fl<16, head_dim>;
    using o_rt_bf    = rt_bf<16, head_dim>;
    using o_sv       = sv_bf<head_dim>;

    struct parsed_instruction {
        int layer_idx;
        int batch_block_idx;
        int kv_head_idx;
        __device__ inline parsed_instruction(typename Config::instruction_t &i) {
            layer_idx       = i[1];
            batch_block_idx = i[2];
            kv_head_idx     = i[3];
        }
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    };

    // ── semaphore accessors ──────────────────────────────────────────────────
    __device__ static inline semaphore &O_arrived(state<Config> &s)             { return s.semaphores()[0]; }
    __device__ static inline semaphore &K_arrived(state<Config> &s, int stage)  { return s.semaphores()[1 + stage * 2]; }
    __device__ static inline semaphore &V_arrived(state<Config> &s, int stage)  { return s.semaphores()[1 + stage * 2 + 1]; }
    __device__ static inline semaphore &K_finished(state<Config> &s, int stage) { return s.semaphores()[1 + NUM_STAGES * 2 + stage * 2]; }
    __device__ static inline semaphore &V_finished(state<Config> &s, int stage) { return s.semaphores()[1 + NUM_STAGES * 2 + stage * 2 + 1]; }

    // ── shmem layout helpers ─────────────────────────────────────────────────
    __device__ static inline void wait_KV_page(state<Config> &s, int stage) {
        // Stage uses pages [stage*PAGES_PER_STAGE .. stage*PAGES_PER_STAGE + K_PAGES + V_PAGES).
        for (int p = 0; p < PAGES_PER_STAGE; p++)
            s.wait_page_ready(s.pid(stage * PAGES_PER_STAGE + p));
    }
    __device__ static inline void finish_KV_page(state<Config> &s, int stage) {
        int count = Config::NUM_CONSUMER_WARPS / ATTN_BATCH_BLOCK_SIZE;
        for (int p = 0; p < PAGES_PER_STAGE; p++)
            s.finish_page(s.pid(stage * PAGES_PER_STAGE + p), count);
    }

    __device__ static inline kv_st &get_K_smem(state<Config> &s, int stage, int batch_idx) {
        // K tiles are in the first K_PAGES pages of the stage.
        char *base = reinterpret_cast<char *>(s.pages[s.pid(stage * PAGES_PER_STAGE)].data);
        return *reinterpret_cast<kv_st *>(base + sizeof(kv_st) * batch_idx);
    }
    __device__ static inline kv_st &get_V_smem(state<Config> &s, int stage, int batch_idx) {
        // V tiles follow K tiles (in their own pages).
        char *base = reinterpret_cast<char *>(s.pages[s.pid(stage * PAGES_PER_STAGE + K_PAGES)].data);
        return *reinterpret_cast<kv_st *>(base + sizeof(kv_st) * batch_idx);
    }

    // Q and O share the same scratch region (same as Hopper).
    // Scratch layout: o_sv[4] per batch item, ATTN_BATCH_BLOCK_SIZE items.
    __device__ static inline q_st &get_Q_smem(state<Config> &s, int batch_idx) {
        return *reinterpret_cast<q_st *>(reinterpret_cast<char *>(s.scratch()) + sizeof(o_sv) * batch_idx * 4);
    }
    __device__ static inline o_sv (&get_O_smem(state<Config> &s, int batch_idx))[4] {
        return *reinterpret_cast<o_sv(*)[4]>(reinterpret_cast<char *>(s.scratch()) + sizeof(o_sv) * batch_idx * 4);
    }

    // Load Q using cp.async (same approach as the original load_Q_async which already used cp.async).
    __device__ static inline void load_Q_async(q_st &dst,
                                               const typename Globals::activations_t &src,
                                               int batch_idx, int q_head_start_idx) {
        using T = typename q_st::dtype;
        constexpr int elem_per_memcpy = sizeof(float4) / sizeof(T);
        constexpr int memcpy_per_row  = head_dim / elem_per_memcpy;

        typename Globals::activations_t::dtype *src_ptr =
            (typename Globals::activations_t::dtype *)&src[coord<>{batch_idx, q_head_start_idx * head_dim}];
        uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
        int lid = warp::laneid();
        int col = (lid % memcpy_per_row) * elem_per_memcpy;
        int base_row = (lid < memcpy_per_row) ? 0 : 1;

        for (int i = 0; i < (GQA_RATIO / 2); i++) {
            int row = base_row + i * 2;
            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::
                "r"(dst.idx(dst_ptr, {row, col})),
                "l"(&src_ptr[row * head_dim + col]) : "memory");
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    // store_4_rows: unchanged from the original (pure register→shmem, no TMA).
    template <ducks::sv::all SV, ducks::rt::all RT>
    __device__ static inline void store_4_rows(SV (&dst)[4], const RT &src) {
        static_assert(RT::rows == 16);
        static_assert(SV::length == src.cols);
        using T2 = typename RT::dtype;
        using U  = typename SV::dtype;
        using U2 = typename base_types::packing<U>::packed_type;
        uint32_t dst_ptr[4];
        for (int i = 0; i < 4; ++i)
            dst_ptr[i] = static_cast<uint32_t>(__cvta_generic_to_shared(&dst[i].data[0]));
        int lid = kittens::laneid();
        if (lid < 16) {
            int lr = lid / 4, lc = lid % 4;
            for (int j = 0; j < src.width; j++) {
                U2 tmp[2];
                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[2]);
                int ci = lc * 2 + j * 16;
                move<U2>::sts(dst_ptr[lr] + sizeof(U) * ci,     tmp[0]);
                move<U2>::sts(dst_ptr[lr] + sizeof(U) * (ci+8), tmp[1]);
            }
        }
    }

    template <ducks::rt::row_layout RT>
    __device__ static inline void right_fill(RT &dst, const RT &src, int col_idx,
                                             typename base_types::packing<typename RT::dtype>::unpacked_type val = 0) {
        if (col_idx >= dst.cols) return;
        for (int i = 0; i < dst.height; i++)
            for (int j = 0; j < dst.width; j++)
                for (int k = 0; k < dst.packed_per_tile; k++) {
                    auto &d = dst.tiles[i][j].data[k];
                    auto &sv = src.tiles[i][j].data[k];
                    int cx = (j * dst.tile_size_col) + ((k / 2) * 8) + ((warp::laneid() % 4) * 2);
                    int cy = cx + 1;
                    d.x = (cx >= col_idx) ? val : sv.x;
                    d.y = (cy >= col_idx) ? val : sv.y;
                }
    }

    // ── controller ───────────────────────────────────────────────────────────
    struct controller {
        static __device__ int release_lid(const Globals &g,
                                          typename Config::instruction_t &ins, int &query) {
            // Cycle through stage page groups.
            return (query % NUM_STAGES) * PAGES_PER_STAGE;
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
            init_semaphore(O_arrived(s), 0, ATTN_BATCH_BLOCK_SIZE);
            for (int i = 0; i < NUM_STAGES; i++) {
                init_semaphore(K_arrived(s, i),  0, ATTN_BATCH_BLOCK_SIZE);
                init_semaphore(V_arrived(s, i),  0, ATTN_BATCH_BLOCK_SIZE);
                init_semaphore(K_finished(s, i), 0, ATTN_BATCH_BLOCK_SIZE);
                init_semaphore(V_finished(s, i), 0, ATTN_BATCH_BLOCK_SIZE);
            }
            return 1 + 4 * NUM_STAGES;
        }
    };

    // ── loader ───────────────────────────────────────────────────────────────
    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            int lid      = warp::laneid();
            int seq_len  = g.pos_id + 1;
            int total_blocks = (seq_len + kv_block_size - 1) / kv_block_size;

            if (lid < ATTN_BATCH_BLOCK_SIZE) {
                int batch_block_idx = (inst.batch_block_idx * ATTN_BATCH_BLOCK_SIZE)
                                    / Globals::matmul_batch_block_size;

                for (int i = 0; i < total_blocks; ++i) {
                    int stage = i % NUM_STAGES;
                    kv_st &K_smem = get_K_smem(s, stage, lid);
                    kv_st &V_smem = get_V_smem(s, stage, lid);

                    if (i >= NUM_STAGES) {
                        wait(K_finished(s, stage), (i / NUM_STAGES - 1) % 2);
                        wait(V_finished(s, stage), (i / NUM_STAGES - 1) % 2);
                    } else {
                        wait_KV_page(s, stage);
                    }

                    // Wait for QKV write to KV cache.
                    if (i == 0) {
                        while (*(volatile int *)&g.Bar[{inst.layer_idx, OPCODE_QKV_RopeAppend - 1,
                               batch_block_idx, Globals::num_attention_heads + inst.kv_head_idx}] < 1)
                            __nanosleep(20);
                    }

                    // ── cp.async load K ──────────────────────────────────────
                    // K cache: [batch_entry, seq_block, kv_head, 0]
                    int batch_entry = (int)g.batch_size * inst.layer_idx
                                    + inst.batch_block_idx * ATTN_BATCH_BLOCK_SIZE + lid;
                    warp::load_async(K_smem, g.k_cache, {batch_entry, i, inst.kv_head_idx, 0});

                    if (i == 0) {
                        while (*(volatile int *)&g.Bar[{inst.layer_idx, OPCODE_QKV_RopeAppend - 1,
                               batch_block_idx, (int)Globals::num_attention_heads
                               + (int)Globals::num_kv_heads + inst.kv_head_idx}] < 1)
                            __nanosleep(20);
                    }

                    // ── cp.async load V ──────────────────────────────────────
                    warp::load_async(V_smem, g.v_cache, {batch_entry, i, inst.kv_head_idx, 0});

                    asm volatile("cp.async.commit_group;\n" ::: "memory");
                    asm volatile("cp.async.wait_all;\n"     ::: "memory");

                    // Signal consumers.
                    arrive(K_arrived(s, stage));
                    arrive(V_arrived(s, stage));
                }
            } else {
                // Release unused pages (pages beyond NUM_STAGES).
                int page_idx = lid - ATTN_BATCH_BLOCK_SIZE;
                if (page_idx >= (int)min(NUM_STAGES * PAGES_PER_STAGE, (int)Config::NUM_PAGES)) {
                    int unused = s.pid(page_idx);
                    s.wait_page_ready(unused);
                    s.finish_page(unused, Config::NUM_CONSUMER_WARPS);
                }
            }
        }
    };

    // ── launcher: no-op on sm89 ──────────────────────────────────────────────
    struct launcher {
        static __device__ void run(const Globals &g, state<Config> &s) {
#ifdef KITTENS_BLACKWELL
            if (warp::laneid() == 0) {
                s.wait_tensor_ready();
                arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
            }
#endif
        }
    };

    // ── consumer: same flash-attention math as Hopper (already warp-scope mma) ──
    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            int wid = group<Config::NUM_CONSUMER_WARPS>::warpid();

            if (wid < ATTN_BATCH_BLOCK_SIZE) {
                int batch_idx      = inst.batch_block_idx * ATTN_BATCH_BLOCK_SIZE + wid;
                int batch_block_idx = batch_idx / Globals::matmul_batch_block_size;
                int q_head_start   = inst.kv_head_idx * GQA_RATIO;

                // Wait for all GQA heads to be written.
                for (int i = 0; i < GQA_RATIO; i++) {
                    while (*(volatile int *)&g.Bar[{inst.layer_idx, OPCODE_QKV_RopeAppend - 1,
                           batch_block_idx, inst.kv_head_idx * GQA_RATIO + i}] < 1)
                        __nanosleep(20);
                }

                q_st &Q_smem = get_Q_smem(s, wid);
                load_Q_async(Q_smem, g.q_post_rope, batch_idx, q_head_start);

                q_rt  Q_reg;  o_rt O_reg;
                k_rt  K_reg;  v_rt V_reg;
                attn_fl_rt attn_fl;  attn_bf_rt attn_bf;
                max_vec_rv max_vec, scaled_max, last_scaled_max, diff_scaled_max;
                norm_vec_rv norm_vec;

                warp::neg_infty(max_vec);
                warp::zero(last_scaled_max);
                warp::zero(norm_vec);
                warp::zero(O_reg);

                float softmax_temp = g.attn_scale * 1.44269504089f;

                warp::load_async_wait();
                warp::load(Q_reg, Q_smem);

                int seq_len    = g.pos_id + 1;
                int total_blks = (seq_len + kv_block_size - 1) / kv_block_size;

                for (int i = 0; i < total_blks; i++) {
                    int stage = i % NUM_STAGES;
                    kv_st &K_smem = get_K_smem(s, stage, wid);
                    kv_st &V_smem = get_V_smem(s, stage, wid);

                    warp::zero(attn_fl);
                    warp::wait(K_arrived(s, stage), (i / NUM_STAGES) % 2);
                    warp::load(K_reg, K_smem);
                    warp::mma_ABt(attn_fl, Q_reg, K_reg, attn_fl);
                    warp::sync();
                    warp::arrive(K_finished(s, stage));

                    if ((i + 1) * kv_block_size > seq_len)
                        right_fill(attn_fl, attn_fl, seq_len % kv_block_size, -999999999999.f);

                    warp::row_max(max_vec, attn_fl, max_vec);
                    warp::mul(attn_fl, attn_fl, softmax_temp);
                    warp::mul(scaled_max, max_vec, softmax_temp);
                    warp::sub_row(attn_fl, attn_fl, scaled_max);
                    warp::exp2(attn_fl, attn_fl);
                    warp::sub(diff_scaled_max, last_scaled_max, scaled_max);
                    warp::exp2(diff_scaled_max, diff_scaled_max);
                    warp::mul_row(O_reg, O_reg, diff_scaled_max);

                    warp::wait(V_arrived(s, stage), (i / NUM_STAGES) % 2);
                    warp::load(V_reg, V_smem);
                    warp::copy(attn_bf, attn_fl);
                    warp::mma_AB(O_reg, attn_bf, V_reg, O_reg);
                    warp::sync();
                    warp::arrive(V_finished(s, stage));

                    warp::mul(norm_vec, norm_vec, diff_scaled_max);
                    warp::row_sum(norm_vec, attn_fl, norm_vec);
                    warp::copy(last_scaled_max, scaled_max);

                    if (total_blks - i <= NUM_STAGES && warp::laneid() == 0)
                        finish_KV_page(s, stage);
                }

                warp::div_row(O_reg, O_reg, norm_vec);
                o_rt_bf O_bf;
                warp::copy(O_bf, O_reg);
                o_sv (&O_smem)[4] = get_O_smem(s, wid);
                store_4_rows(O_smem, O_bf);
                warp::sync();
                warp::arrive(O_arrived(s));
            }
        }
    };

    // ── storer ───────────────────────────────────────────────────────────────
    struct storer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            int lid           = warp::laneid();
            int q_head_start  = inst.kv_head_idx * GQA_RATIO;

            wait(O_arrived(s), 0);

            if (lid < ATTN_BATCH_BLOCK_SIZE * GQA_RATIO) {
                int batch_in_block = lid / GQA_RATIO;
                int head_in_group  = lid % GQA_RATIO;
                o_sv (&O_smem)[4]  = get_O_smem(s, batch_in_block);
                // Plain warp::store — warp-scope, no TMA.
                // Each lane (0..15) writes one sv_bf<head_dim> to a distinct attn_out row.
                // Only GQA_RATIO lanes per batch item participate:
                // attn_out shape: [1, 1, batch, num_q_heads * head_dim]
                int out_batch = inst.batch_block_idx * ATTN_BATCH_BLOCK_SIZE + batch_in_block;
                int out_head  = q_head_start + head_in_group;
                if (laneid() < ATTN_BATCH_BLOCK_SIZE * GQA_RATIO)
                    warp::store(g.attn_out, O_smem[head_in_group],
                                {out_batch, out_head});
            }

            __syncwarp();
            asm volatile("{fence.acq_rel.gpu;}");

            if (lid == 0) {
                int bb = (inst.batch_block_idx * ATTN_BATCH_BLOCK_SIZE) / Globals::matmul_batch_block_size;
                atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1, bb, 0}], ATTN_BATCH_BLOCK_SIZE);
            }
        }
    };
};

} // namespace kittens::prototype::vm
