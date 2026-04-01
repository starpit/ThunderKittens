// sm89 port of qkv_rope_append.cu
// QKV projection + RoPE rotation + KV cache write.
// Changes vs Hopper:
//   - matmul_pipeline → matmul_pipeline_sm89
//   - rope_cos/sin loaded with cp.async into a page (not scratch+TMA)
//   - consumer uses rt_fl register acc (no tensor_alloc)
//   - storer uses warp::store (no warp::tma::store_async)

#include "llama_sm89.cuh"
#include "matmul_pipeline_sm89.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm {

using globals = llama_sm89_globals;
using config  = llama_sm89_config;

struct qkv_gmem_waiter_sm89;

template <typename Config = config, typename Globals = globals>
struct qkv_rope_append {
    static constexpr int opcode        = OPCODE_QKV_RopeAppend;
    // Head-block boundaries in output-column-block units.
    // Each col block is OUT_BLOCK=64 cols = 1 head (since head_dim=64 for 1B or 128 for 8B).
    // For LLaMA 1B:  Q cols = 0..31, K cols = 32..39, V cols = 40..47
    // For LLaMA 8B:  Q cols = 0..31, K cols = 32..39, V cols = 40..47 (same in head count)
    static constexpr int K_BLOCK_START = Globals::num_attention_heads;
    static constexpr int V_BLOCK_START = Globals::num_attention_heads + Globals::num_kv_heads;
    static constexpr int NUM_ITERS     = Globals::hidden_dim / SM89_PIPELINE_K_DIM;
    static constexpr int OUT_BLOCK     = Globals::matmul_out_block_size;
    static constexpr int BATCH_BLOCK   = Globals::matmul_batch_block_size;

    // Rope vectors: sv_fl<head_dim> each.  Store cos+sin in one extra page (page 6 when GEMM uses 0-5).
    static constexpr int ROPE_PAGE = config::NUM_PAGES - 1;  // use last page for rope vectors

    struct parsed_instruction {
        int layer, row, col;
        __device__ inline parsed_instruction(typename Config::instruction_t &i) {
            layer = i[1]; row = i[2]; col = i[3];
        }
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    };

    using pipeline = matmul_pipeline_sm89<Config, Globals, parsed_instruction,
                                          qkv_gmem_waiter_sm89,
                                          &Globals::rms_rope_intermediates,
                                          &Globals::qkv_weights,
                                          NUM_ITERS>;
    using acc_rt = typename pipeline::acc_rt;

    __device__ static inline semaphore &rope_arrived(state<Config> &s) {
        return s.semaphores()[pipeline::SEM_COUNT];
    }

    struct controller {
        static __device__ int release_lid(const Globals &g, typename Config::instruction_t &ins, int &q) {
            return pipeline::release_lid(g, ins, q);
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
            init_semaphore(rope_arrived(s), 1);
            return pipeline::init_semaphores(s) + 1;
        }
    };

    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            pipeline::loader_loop(s, g, inst.layer);
            warp::sync();

            // Load rope cos/sin only for Q and K cols (not V).
            if ((inst.col * 2) < V_BLOCK_START && laneid() == 0) {
                // Rope page holds: sv_fl<head_dim> cos + sv_fl<head_dim> sin.
                using rope_vec = sv_fl<Globals::head_dim>;
                int pid = s.pid(ROPE_PAGE);
                s.wait_page_ready(pid);
                auto *rope_base = reinterpret_cast<char *>(s.pages[pid].data);
                rope_vec &rope_cos = *reinterpret_cast<rope_vec *>(rope_base);
                rope_vec &rope_sin = *reinterpret_cast<rope_vec *>(rope_base + sizeof(rope_vec));

                warp::load_async(rope_cos, g.rope_cos, {(int)g.pos_id, 0});
                warp::load_async(rope_sin, g.rope_sin, {(int)g.pos_id, 0});
                asm volatile("cp.async.commit_group;\n" ::: "memory");
                asm volatile("cp.async.wait_all;\n"     ::: "memory");
                arrive(rope_arrived(s));
            }
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g, state<Config> &s) {
            pipeline::launcher_loop(s, g);
        }
    };

    struct storer { static __device__ void run(const Globals &g, state<Config> &s) {} };

    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            wait(pipeline::outputs_arrived(s), 0);

            acc_rt acc;
            pipeline::consumer_loop(s, g, acc);

            // Each consumer warp holds OUT_BLOCK output columns for
            // rows [warpid()*16 .. (warpid()+1)*16) of the QKV result.
            int global_row = inst.row * BATCH_BLOCK + warpid() * 16;
            bool is_q_block = (inst.col < K_BLOCK_START);
            bool is_k_block = (inst.col >= K_BLOCK_START) && (inst.col < V_BLOCK_START);
            bool is_v_block = (inst.col >= V_BLOCK_START);
            bool needs_rope = is_q_block || is_k_block;

            // Apply RoPE if needed.
            if (needs_rope) {
                wait(rope_arrived(s), 0);
                using rope_vec = sv_fl<Globals::head_dim>;
                auto *rope_base = reinterpret_cast<char *>(s.pages[s.pid(ROPE_PAGE)].data);
                rope_vec &cos_sv = *reinterpret_cast<rope_vec *>(rope_base);
                rope_vec &sin_sv = *reinterpret_cast<rope_vec *>(rope_base + sizeof(rope_vec));

                rv_fl<OUT_BLOCK, ducks::rv_layout::align> cos_rv, sin_rv;
                warp::load(cos_rv, cos_sv);
                warp::load(sin_rv, sin_sv);

                // RoPE rotation: applied element-by-element in the packed (float2) layout.
                // TK stores rt_fl data as packed float2 tiles. The two elements of each pair
                // are the (real, imag) = (even_idx, odd_idx) components for RoPE.
                #pragma unroll
                for (int r = 0; r < acc_rt::height; r++) {
                    #pragma unroll
                    for (int c = 0; c < acc_rt::width; c++) {
                        #pragma unroll
                        for (int k = 0; k < acc_rt::packed_per_tile; k++) {
                            float2 &v = acc.tiles[r][c].data[k];
                            // Corresponding cos/sin pair index
                            int cos_idx = c * acc_rt::tile_size_col / 2 + k;
                            float c_ = cos_rv.data[cos_idx];
                            float s_ = sin_rv.data[cos_idx];
                            float x  = v.x * c_ - v.y * s_;
                            float y  = v.x * s_ + v.y * c_;
                            v.x = x;
                            v.y = y;
                        }
                    }
                }
            }

            // Convert to bf16 and store.
            rt_bf<16, OUT_BLOCK> out_bf;
            warp::copy(out_bf, acc);

            if (is_q_block) {
                warp::store(g.q_post_rope, out_bf, {global_row / 16, inst.col});
            } else if (is_k_block) {
                // Store into KV cache at [batch_idx, pos_id, kv_head_idx, 0]
                int kv_head_idx = inst.col - K_BLOCK_START;
                int seq_entry   = (int)g.batch_size * 0 /*layer_idx=0 here, loop outside*/ +
                                  global_row / 16;
                // Note: the layer index is handled by the instruction dispatcher.
                // Here we write to k_cache using the token's position.
                // For simplicity, write once at pos_id.
                warp::store(g.k_cache, out_bf, {seq_entry, (int)g.pos_id, kv_head_idx, 0});
            } else {
                int kv_head_idx = inst.col - V_BLOCK_START;
                int seq_entry   = (int)g.batch_size * 0 + global_row / 16;
                warp::store(g.v_cache, out_bf, {seq_entry, (int)g.pos_id, kv_head_idx, 0});
            }

            __threadfence();

            // Release the rope page.
            if (needs_rope && laneid() == 0)
                s.finish_page(s.pid(ROPE_PAGE), Config::NUM_CONSUMER_WARPS);

            warp::sync();
            if (laneid() == 0) {
                int start_bar     = (inst.col * OUT_BLOCK) / Globals::head_dim;
                int heads_per_col = OUT_BLOCK / Globals::head_dim;
                for (int i = 0; i < heads_per_col; i++)
                    atomicAdd(&g.Bar[{inst.layer, opcode - 1, inst.row, start_bar + i}], 1);
            }
        }
    };
};

struct qkv_gmem_waiter_sm89 {
    template <typename Cfg, typename G, typename Inst>
    static __device__ inline void gmem_wait(const G &g, state<Cfg> &s, Inst &inst) {
        while (*(volatile int *)&g.Bar[{inst.layer, OPCODE_AttnNorm - 1, inst.row, 0}]
               < (int)G::matmul_batch_block_size)
            __nanosleep(20);
    }
};

} // namespace kittens::prototype::vm
