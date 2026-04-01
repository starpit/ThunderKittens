// Minimal compilation test for sm89 KVM — no pybind, no Python.
// Just checks that the sm89 ops compile on sm_89.
// Build: see Makefile.compile target.

#include "llama_sm89.cuh"

#include "rms_norm_sm89.cu"
#include "qkv_rope_append_sm89.cu"
#include "attention_decode_sm89.cu"
#include "matmul_adds_sm89.cu"
#include "gate_silu_sm89.cu"
#include "up_matmul_sm89.cu"
#include "lm_head_sm89.cu"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

using G = llama_sm89_globals;
using C = llama_sm89_config;

using attn_norm_op    = attn_norm    <C, G>;
using qkv_op          = qkv_rope_append<C, G>;
using attn_op         = attention_decode<C, G>;
using o_proj_op       = o_proj       <C, G>;
using mlp_norm_op     = mlp_norm     <C, G>;
using gate_silu_op    = gate_silu    <C, G>;
using up_matmul_op    = up_matmul    <C, G>;
using downproj_op     = downproj     <C, G>;
using lm_head_norm_op = lm_head_norm <C, G>;
using lm_head_op      = lm_head      <C, G>;

// Instantiate the KVM kernel.
using MyKVM = kvm<C, G,
    attn_norm_op, qkv_op, attn_op, o_proj_op,
    mlp_norm_op, gate_silu_op, up_matmul_op, downproj_op,
    lm_head_norm_op, lm_head_op>;

// Host entry point (not called, just ensures the kernel is instantiated).
void launch_kvm(G globals, cudaStream_t stream) {
    auto kernel = kittens::prototype::vm::kvm_kernel<C, G,
        attn_norm_op, qkv_op, attn_op, o_proj_op,
        mlp_norm_op, gate_silu_op, up_matmul_op, downproj_op,
        lm_head_norm_op, lm_head_op>;

    // Set max dynamic shmem.
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         C::DYNAMIC_SHARED_MEMORY);

    kernel<<<globals.grid(), globals.block(), globals.dynamic_shared_memory(), stream>>>(globals);
}

int main() {
    // Compile-only test: just print config info.
    printf("sm89 KVM config:\n");
    printf("  PAGE_SIZE         = %d bytes\n", C::PAGE_SIZE);
    printf("  NUM_PAGES         = %d\n",       C::NUM_PAGES);
    printf("  STATIC_SHMEM      = %d bytes\n", C::STATIC_SHARED_MEMORY);
    printf("  DYNAMIC_SHMEM     = %d bytes\n", C::DYNAMIC_SHARED_MEMORY);
    printf("  hidden_dim        = %d\n",        G::hidden_dim);
    printf("  matmul_out_block  = %d\n",        G::matmul_out_block_size);
    printf("  SM89_PIPELINE_K   = %d\n",        SM89_PIPELINE_K_DIM);
    printf("  attn NUM_STAGES   = %d\n",        attention_decode<C,G>::NUM_STAGES);
    return 0;
}
