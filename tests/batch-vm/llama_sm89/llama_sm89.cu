#include "llama_sm89.cuh"

#include "rms_norm_sm89.cu"
#include "qkv_rope_append_sm89.cu"
#include "attention_decode_sm89.cu"
#include "matmul_adds_sm89.cu"
#include "gate_silu_sm89.cu"
#include "up_matmul_sm89.cu"
#include "lm_head_sm89.cu"

#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

// Concrete op types for the sm89 KVM.
using attn_norm_op     = attn_norm    <llama_sm89_config, llama_sm89_globals>;
using qkv_op           = qkv_rope_append<llama_sm89_config, llama_sm89_globals>;
using attn_op          = attention_decode<llama_sm89_config, llama_sm89_globals>;
using o_proj_op        = o_proj       <llama_sm89_config, llama_sm89_globals>;
using mlp_norm_op      = mlp_norm     <llama_sm89_config, llama_sm89_globals>;
using gate_silu_op     = gate_silu    <llama_sm89_config, llama_sm89_globals>;
using up_matmul_op     = up_matmul    <llama_sm89_config, llama_sm89_globals>;
using downproj_op      = downproj     <llama_sm89_config, llama_sm89_globals>;
using lm_head_norm_op  = lm_head_norm <llama_sm89_config, llama_sm89_globals>;
using lm_head_op       = lm_head      <llama_sm89_config, llama_sm89_globals>;

PYBIND11_MODULE(kvm_llama_sm89, m) {
    m.doc() = "KVM LLaMA megakernel for sm89 (L40S / RTX 4090)";

    kittens::py::bind_kernel<kvm<llama_sm89_config,
        llama_sm89_globals,
        attn_norm_op,
        qkv_op,
        attn_op,
        o_proj_op,
        mlp_norm_op,
        gate_silu_op,
        up_matmul_op,
        downproj_op,
        lm_head_norm_op,
        lm_head_op
    >>(m, "kvm_llama_sm89",
        &llama_sm89_globals::Bar,
        &llama_sm89_globals::instructions,
        &llama_sm89_globals::timings,

        &llama_sm89_globals::qkv_weights,
        &llama_sm89_globals::attn_norm_weights,
        &llama_sm89_globals::o_weights,
        &llama_sm89_globals::mlp_norm_weights,

        &llama_sm89_globals::up_weights,
        &llama_sm89_globals::gate_weights,
        &llama_sm89_globals::down_weights,

        &llama_sm89_globals::lm_head_norm_weights,
        &llama_sm89_globals::lm_head_weights,

        &llama_sm89_globals::k_cache,
        &llama_sm89_globals::v_cache,

        &llama_sm89_globals::rope_cos,
        &llama_sm89_globals::rope_sin,

        &llama_sm89_globals::hidden_states,
        &llama_sm89_globals::rms_rope_intermediates,
        &llama_sm89_globals::rms_gate_intermediates,

        &llama_sm89_globals::q_post_rope,
        &llama_sm89_globals::attn_out,
        &llama_sm89_globals::silu_out,

        &llama_sm89_globals::rms_lm_head_intermediates,
        &llama_sm89_globals::logits,
        &llama_sm89_globals::pos_id,
        &llama_sm89_globals::attn_scale,
        &llama_sm89_globals::rms_norm_eps,
        &llama_sm89_globals::batch_size
    );
}
