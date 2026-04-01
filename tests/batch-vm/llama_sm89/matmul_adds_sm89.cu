// sm89 port of matmul_adds.cu (O projection and down projection with residual add).
// Replaces:
//   - s.tensor_alloc + tt<> + kittens::mm  →  warp::mma on rt_fl accumulator
//   - warp::tma::store_add_async           →  load existing + add + store (plain)
//   - tma::store_async_read_wait           →  __threadfence_block()

#include "llama_sm89.cuh"
#include "matmul_pipeline_sm89.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm {

using globals = llama_sm89_globals;
using config  = llama_sm89_config;

template <
    auto InputActivationsPtr,
    auto WeightsPtr,
    auto OutputActivationsPtr,
    int  iters,
    int  _opcode,
    typename gmem_waiter,
    typename Config  = config,
    typename Globals = globals>
struct MatMulAddOp_sm89 {
    static constexpr int opcode    = _opcode;
    static constexpr int BATCH_BLOCK = Globals::matmul_batch_block_size;
    static constexpr int OUT_BLOCK   = Globals::matmul_out_block_size;

    struct parsed_instruction {
        int layer;
        int row;   // batch block index
        int col;   // output block index
        __device__ inline parsed_instruction(typename Config::instruction_t &instruction) {
            layer = instruction[1];
            row   = instruction[2];
            col   = instruction[3];
        }
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    };

    using pipeline = matmul_pipeline_sm89<Config, Globals, parsed_instruction, gmem_waiter,
                                          InputActivationsPtr, WeightsPtr, iters>;
    using acc_rt   = typename pipeline::acc_rt;

    struct controller {
        static __device__ int release_lid(const Globals &g,
                                          typename Config::instruction_t &instruction,
                                          int &query) {
            return pipeline::release_lid(g, instruction, query);
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
            return pipeline::init_semaphores(s);
        }
    };

    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            pipeline::loader_loop(s, g, inst.layer);
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g, state<Config> &s) {
            pipeline::launcher_loop(s, g);
        }
    };

    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};

            // Wait for all K-tiles to be done (outputs_arrived signals matmul complete).
            wait(pipeline::outputs_arrived(s), 0);

            // Run warp-scope GEMM accumulation.
            acc_rt acc;
            pipeline::consumer_loop(s, g, acc);

            // Convert fp32 accumulator → bf16 and store to output (with residual add).
            // The output GlobalLayout is activations_t; we address:
            //   batch rows: [inst.row * BATCH_BLOCK + warpid()*16 .. +16)
            //   output cols: [inst.col * OUT_BLOCK .. +OUT_BLOCK)
            auto &OutputActivations = g.*OutputActivationsPtr;

            // Warp-level: load existing residual, add, store.
            rt_bf<16, OUT_BLOCK> existing;
            int global_row = inst.row * BATCH_BLOCK + warpid() * 16;
            warp::load(existing, OutputActivations, {global_row / 16, inst.col});

            // Cast accumulator to bf16 in-register, then add residual.
            rt_bf<16, OUT_BLOCK> acc_bf;
            warp::copy(acc_bf, acc);
            warp::add(acc_bf, acc_bf, existing);
            warp::store(OutputActivations, acc_bf, {global_row / 16, inst.col});
            __threadfence();   // flush to L2 so dependent warps/SMs see the update

            // Signal completion to scheduler.
            warp::sync();
            if (laneid() == 0) {
                // Use atomicAdd on the barrier for this (layer, opcode, row) tuple.
                // The storer in the original design did this; here the consumer does it
                // since there is no separate store-TMA step.
                atomicAdd(&g.Bar[{inst.layer, opcode - 1, inst.row, 0}], 1);
            }
        }
    };

    // No separate storer needed: consumer writes directly via warp::store.
    struct storer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            // Nothing: consumer handled the output write + barrier signal.
        }
    };
};

// ─── concrete ops ────────────────────────────────────────────────────────────

struct o_proj_gmem_waiter {
    template <typename Cfg, typename G, typename Inst>
    static __device__ inline void gmem_wait(const G &g, state<Cfg> &s, Inst &inst) {
        while (*(volatile int *)&g.Bar[{inst.layer, OPCODE_GQA_AttentionDecode - 1, inst.row, 0}]
               < (int)(G::matmul_batch_block_size * G::num_kv_heads))
            __nanosleep(20);
    }
};
template <typename Config, typename Globals>
struct o_proj : MatMulAddOp_sm89<
    &Globals::attn_out,
    &Globals::o_weights,
    &Globals::hidden_states,
    Globals::hidden_dim / SM89_PIPELINE_K_DIM,
    OPCODE_O_ProjResidual,
    o_proj_gmem_waiter,
    Config, Globals> {};

struct downproj_gmem_waiter {
    template <typename Cfg, typename G, typename Inst>
    static __device__ inline void gmem_wait(const G &g, state<Cfg> &s, Inst &inst) {
        while (*(volatile int *)&g.Bar[{inst.layer, OPCODE_UpMatmul - 1, inst.row, 0}]
               < (int)(G::intermediate_dim / G::matmul_out_block_size))
            __nanosleep(20);
    }
};
template <typename Config, typename Globals>
struct downproj : MatMulAddOp_sm89<
    &Globals::silu_out,
    &Globals::down_weights,
    &Globals::hidden_states,
    Globals::intermediate_dim / SM89_PIPELINE_K_DIM,
    OPCODE_DownProjResidual,
    downproj_gmem_waiter,
    Config, Globals> {};

} // namespace kittens::prototype::vm
