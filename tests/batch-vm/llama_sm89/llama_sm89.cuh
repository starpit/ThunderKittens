#pragma once

// sm89 (L40S/RTX 4090) port of the KVM Llama megakernel.
// Key differences from llama_official (Blackwell/Hopper):
//   - No TMA: all global->shared loads use cp.async (group::load_async)
//   - No WGMMA / tensor allocator: GEMM uses warp-scope mma.sync via warp::mma
//   - Smaller shmem budget (~91KB dynamic vs 228KB on H100)
//   - PAGE_SIZE = 8192 (8KB), NUM_PAGES = 10
//   - PIPELINE_K_DIM = 32 (vs 64), matmul_out_block_size = 64 (vs 256)
//   - No cluster launches (sm89 has no cluster support)

#include "kittens.cuh"
#include "vm/vm.cuh"
#include <iostream>

#define OPCODE_AttnNorm         1
#define OPCODE_QKV_RopeAppend   2
#define OPCODE_GQA_AttentionDecode 3
#define OPCODE_O_ProjResidual   4
#define OPCODE_MlpNorm          5
#define OPCODE_GateSiLU         6
#define OPCODE_UpMatmul         7
#define OPCODE_DownProjResidual 8
#define OPCODE_LM_HeadNorm      9
#define OPCODE_LM_Head          10

// Model size selection — default to 1B for easier sm89 testing.
// Override with -DUSE_LLAMA_SM89_8B.
#if defined(USE_LLAMA_SM89_8B)
#  define SM89_NUM_LAYERS              32
#  define SM89_HIDDEN_DIM            4096
#  define SM89_INTERMEDIATE_DIM     14336
#  define SM89_HEAD_DIM               128
#  define SM89_NUM_ATTENTION_HEADS     32
#  define SM89_NUM_KV_HEADS             8
#else
// LLaMA 1B defaults
#  define SM89_NUM_LAYERS              16
#  define SM89_HIDDEN_DIM            2048
#  define SM89_INTERMEDIATE_DIM      8192
#  define SM89_HEAD_DIM                64
#  define SM89_NUM_ATTENTION_HEADS     32
#  define SM89_NUM_KV_HEADS             8
#endif

#define SM89_KV_BLOCK_SIZE          16
// Tile sizes chosen to fit in sm89 shmem budget.
// Output block: 64 cols per SM per dispatch.
// Batch block: 256 rows per SM per dispatch.
#define SM89_MATMUL_OUT_BLOCK_SIZE  64
#define SM89_MATMUL_BATCH_BLOCK_SIZE 256
#define SM89_SM_COUNT               142   // L40S SM count

namespace kittens::prototype::vm {

// ─────────────────────────────────────────────────────────────────────────────
// sm89 KVM config
// ─────────────────────────────────────────────────────────────────────────────
// Shmem budget (L40S, sm89):
//   cudaFuncAttributeMaxDynamicSharedMemorySize → 99KB opt-in
//   SCRATCH_BYTES = 4096
//   STATIC_SHARED_MEMORY = 512 + 2*(4096 + (32+128)*4 + 32*8) = 10496 bytes
//   DYNAMIC_SHARED_MEMORY = 99*1024 - 10496 = 90880 bytes
//   PAGE_SIZE = 8192  → NUM_PAGES = 90880 / 8192 = 11 → use 10 conservatively
//
// GEMM shmem usage per stage:
//   a_smem = st_bf<256, 32> = 16 KB = 2 pages
//   b_smem = st_bf<64,  32> =  4 KB = 1 page  (padded to 1 page)
//   per stage = 3 pages; 2 stages = 6 pages
//   remaining = 4 pages for rms_norm, attention, etc.
struct llama_sm89_config {
    static constexpr int INSTRUCTION_PIPELINE_STAGES       = 2;
    static constexpr int INSTRUCTION_PIPELINE_STAGES_BITS  = 1;
    static constexpr int INSTRUCTION_WIDTH                  = 32;
    using instruction_t = int[INSTRUCTION_WIDTH];
    static constexpr int TIMING_WIDTH                       = 128;
    using timing_t = int[TIMING_WIDTH];
    static constexpr int DYNAMIC_SEMAPHORES                 = 32;

    static constexpr int NUM_CONSUMER_WARPS  = 16;
    static constexpr int NUM_WARPS           = 4 + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS         = NUM_WARPS * ::kittens::WARP_THREADS;
    static constexpr int NUM_BLOCKS          = 1;
    static constexpr int CLUSTER_BLOCKS      = 1;  // no cluster on sm89

    static constexpr int SCRATCH_BYTES = 4096;
    static constexpr int MAX_SHARED_MEMORY = 99 * 1024;  // opt-in to 99 KB
    static constexpr int STATIC_SHARED_MEMORY =
        512 + INSTRUCTION_PIPELINE_STAGES * (SCRATCH_BYTES + (INSTRUCTION_WIDTH + TIMING_WIDTH) * 4 + DYNAMIC_SEMAPHORES * 8);
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int PAGE_SIZE = 8192;
    static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;
    static_assert(NUM_PAGES >= 10, "Need at least 10 pages for sm89 KVM");

    static constexpr bool TIMING_RECORD_ENABLED        = false;
    static constexpr bool GMEM_SPIN_LOOP_SLEEP_NANOS   = 20;

    // Register budgets: no setmaxnreg on sm89, so these are advisory only
    // (the guards in vm.cuh make them no-ops).
    static constexpr int CONSUMER_REGISTERS     = 128;
    static constexpr int NON_CONSUMER_REGISTERS = 64;
};

// ─────────────────────────────────────────────────────────────────────────────
// Globals struct
// ─────────────────────────────────────────────────────────────────────────────
template <int _num_hidden_layers, int _hidden_dim, int _intermediate_dim,
          int _head_dim, int _num_attention_heads, int _num_kv_heads,
          int _kv_block_size, int _matmul_out_block_size, int _matmul_batch_block_size,
          int _sm_count>
struct sm89_globals_t {
    constexpr static int num_hidden_layers       = _num_hidden_layers;
    constexpr static int hidden_dim              = _hidden_dim;
    constexpr static int intermediate_dim        = _intermediate_dim;
    constexpr static int head_dim                = _head_dim;
    constexpr static int num_attention_heads     = _num_attention_heads;
    constexpr static int num_kv_heads            = _num_kv_heads;
    constexpr static int kv_block_size           = _kv_block_size;
    constexpr static int matmul_out_block_size   = _matmul_out_block_size;
    constexpr static int matmul_batch_block_size = _matmul_batch_block_size;
    constexpr static int sm_count                = _sm_count;

    constexpr static int num_output_blocks = hidden_dim / matmul_out_block_size;

    using config = llama_sm89_config;

    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout      = ::kittens::prototype::vm::timing_layout<config>;

    // Weight layouts: [1, num_layers, hidden_dim / out_block, out_block] packed tiles
    // For sm89 GEMM with PIPELINE_K_DIM=32: tiles are st_bf<out_block, 32>
    using weights_t         = gl<bf16, 1, -1, -1, hidden_dim,       st_bf<matmul_out_block_size, 32>>;
    using weights_big_t     = gl<bf16, 1, -1, -1, intermediate_dim, st_bf<matmul_out_block_size, 32>>;

    // Activation layouts: [1, 1, batch, hidden_dim]
    // sv_bf<head_dim> used for per-head vector loads; sv_bf<hidden_dim> for rms vectors
    using activations_t     = gl<bf16, 1, 1, -1, hidden_dim,
                                 sv_bf<head_dim>, sv_bf<hidden_dim>,
                                 st_bf<matmul_batch_block_size, 32>,   // for GEMM A tiles
                                 st_bf<16, 256>>;                       // for storer staging
    using activations_big_t = gl<bf16, 1, 1, -1, intermediate_dim,
                                 st_bf<matmul_batch_block_size, 32>,
                                 st_bf<16, matmul_out_block_size>>;
    using logits_t          = gl<bf16, 1, 1, -1, -1, st_bf<16, matmul_out_block_size>>;

    using norm_weights_t    = gl<bf16, 1, 1, -1, hidden_dim, sv_bf<hidden_dim>>;
    using rope_table_t      = gl<float, 1, 1, -1, head_dim,  sv_fl<head_dim>>;

    // KV cache: no TMA descriptor on sm89 — use plain gl with sv_bf tiles
    using kv_cache_t        = gl<bf16, -1, -1, num_kv_heads, head_dim,
                                 st_bf<kv_block_size, head_dim>>;

    using barriers          = gl<uint, -1, -1, -1, -1>;

    // vm stuff
    barriers         Bar;
    instruction_layout instructions;
    timing_layout    timings;

    // model weights
    weights_t        qkv_weights;
    norm_weights_t   attn_norm_weights;
    weights_t        o_weights;
    norm_weights_t   mlp_norm_weights;
    weights_t        up_weights;
    weights_t        gate_weights;
    weights_big_t    down_weights;
    norm_weights_t   lm_head_norm_weights;
    weights_t        lm_head_weights;

    // kv cache
    kv_cache_t       k_cache;
    kv_cache_t       v_cache;

    // rope tables
    rope_table_t     rope_cos;
    rope_table_t     rope_sin;

    // activation buffers
    activations_t    hidden_states;
    activations_t    rms_rope_intermediates;
    activations_t    rms_gate_intermediates;
    activations_t    q_post_rope;
    activations_t    attn_out;
    activations_big_t silu_out;
    activations_t    rms_lm_head_intermediates;
    logits_t         logits;

    unsigned int     pos_id;
    float            attn_scale;
    float            rms_norm_eps;
    int              batch_size;

    dim3 grid()                  { return dim3(sm_count); }
    dim3 block()                 { return dim3(config::NUM_THREADS); }
    int  dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

typedef sm89_globals_t<
    SM89_NUM_LAYERS,
    SM89_HIDDEN_DIM,
    SM89_INTERMEDIATE_DIM,
    SM89_HEAD_DIM,
    SM89_NUM_ATTENTION_HEADS,
    SM89_NUM_KV_HEADS,
    SM89_KV_BLOCK_SIZE,
    SM89_MATMUL_OUT_BLOCK_SIZE,
    SM89_MATMUL_BATCH_BLOCK_SIZE,
    SM89_SM_COUNT>
    llama_sm89_globals;

// Forward declarations
template <typename config = llama_sm89_config, typename globals = llama_sm89_globals>
struct attn_norm;

template <typename config = llama_sm89_config, typename globals = llama_sm89_globals>
struct qkv_rope_append;

template <typename config = llama_sm89_config, typename globals = llama_sm89_globals>
struct attention_decode;

template <typename config = llama_sm89_config, typename globals = llama_sm89_globals>
struct o_proj;

template <typename config = llama_sm89_config, typename globals = llama_sm89_globals>
struct mlp_norm;

template <typename config = llama_sm89_config, typename globals = llama_sm89_globals>
struct gate_silu;

template <typename config = llama_sm89_config, typename globals = llama_sm89_globals>
struct up_matmul;

template <typename config = llama_sm89_config, typename globals = llama_sm89_globals>
struct downproj;

template <typename config = llama_sm89_config, typename globals = llama_sm89_globals>
struct lm_head_norm;

template <typename config = llama_sm89_config, typename globals = llama_sm89_globals>
struct lm_head;

} // namespace kittens::prototype::vm
