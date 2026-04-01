// sm89 port of up_matmul.cu
// UpMatmul: X @ up_weights^T, element-wise multiply with silu_out (Hadamard product).

#include "llama_sm89.cuh"
#include "matmul_pipeline_sm89.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm {

using globals = llama_sm89_globals;
using config  = llama_sm89_config;

struct up_matmul_gmem_waiter {
    template <typename Cfg, typename G, typename Inst>
    static __device__ inline void gmem_wait(const G &g, state<Cfg> &s, Inst &inst) {
        // Wait for GateSiLU to write silu_out before we read it.
        while (*(volatile int *)&g.Bar[{inst.layer, OPCODE_GateSiLU - 1, inst.row, 0}]
               < (int)(G::intermediate_dim / G::matmul_out_block_size))
            __nanosleep(20);
    }
};

template <typename Config = config, typename Globals = globals>
struct up_matmul {
    static constexpr int opcode    = OPCODE_UpMatmul;
    static constexpr int BATCH_BLOCK = Globals::matmul_batch_block_size;
    static constexpr int OUT_BLOCK   = Globals::matmul_out_block_size;
    static constexpr int NUM_ITERS   = Globals::hidden_dim / SM89_PIPELINE_K_DIM;

    struct parsed_instruction {
        int layer, row, col;
        __device__ inline parsed_instruction(typename Config::instruction_t &i) {
            layer = i[1]; row = i[2]; col = i[3];
        }
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    };

    using pipeline = matmul_pipeline_sm89<Config, Globals, parsed_instruction, up_matmul_gmem_waiter,
                                          &Globals::rms_gate_intermediates, &Globals::up_weights,
                                          NUM_ITERS>;
    using acc_rt = typename pipeline::acc_rt;

    struct controller {
        static __device__ int release_lid(const Globals &g, typename Config::instruction_t &ins, int &q) {
            return pipeline::release_lid(g, ins, q);
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
            return pipeline::init_semaphores(s);
        }
    };
    struct loader  { static __device__ void run(const Globals &g, state<Config> &s) { parsed_instruction i{s}; pipeline::loader_loop(s, g, i.layer); } };
    struct launcher { static __device__ void run(const Globals &g, state<Config> &s) { pipeline::launcher_loop(s, g); } };
    struct storer  { static __device__ void run(const Globals &g, state<Config> &s) {} };

    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            wait(pipeline::outputs_arrived(s), 0);

            acc_rt acc;
            pipeline::consumer_loop(s, g, acc);

            // Element-wise multiply with gate_silu output (already in silu_out).
            int global_row = inst.row * BATCH_BLOCK + warpid() * 16;
            rt_bf<16, OUT_BLOCK> gate_bf;
            warp::load(gate_bf, g.silu_out, {global_row / 16, inst.col});

            // up_result = acc * gate (Hadamard product), both in fp32 then cast.
            rt_fl<16, OUT_BLOCK> gate_fl;
            warp::copy(gate_fl, gate_bf);

            #pragma unroll
            for (int r = 0; r < acc_rt::height; r++) {
                #pragma unroll
                for (int c = 0; c < acc_rt::width; c++) {
                    #pragma unroll
                    for (int rr = 0; rr < acc_rt::tile_size_row; rr++) {
                        #pragma unroll
                        for (int cc = 0; cc < acc_rt::tile_size_col / 2; cc++) {
                            float2 &a = acc.tiles[r][c].data[rr * (acc_rt::tile_size_col / 2) + cc];
                            float2 &g_ = gate_fl.tiles[r][c].data[rr * (acc_rt::tile_size_col / 2) + cc];
                            a.x *= g_.x;
                            a.y *= g_.y;
                        }
                    }
                }
            }

            // Write result back to silu_out (reuse buffer for down_proj input).
            rt_bf<16, OUT_BLOCK> out_bf;
            warp::copy(out_bf, acc);
            warp::store(g.silu_out, out_bf, {global_row / 16, inst.col});
            __threadfence();

            warp::sync();
            if (laneid() == 0)
                atomicAdd(&g.Bar[{inst.layer, opcode - 1, inst.row, 0}], 1);
        }
    };
};

} // namespace kittens::prototype::vm
