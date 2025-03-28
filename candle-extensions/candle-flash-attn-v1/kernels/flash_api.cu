#include "fmha.h"
#include "fmha_utils.h"

void run_fmha_fwd(Launch_params<FMHA_fprop_params> &launch_params) {
    if (launch_params.params.d <= 32) {
        run_fmha_fwd_hdim32(launch_params);
    } else if (launch_params.params.d <= 64) {
        run_fmha_fwd_hdim64(launch_params);
    } else if (launch_params.params.d <= 128) {
        run_fmha_fwd_hdim128(launch_params);
    }
}

extern "C" void run_mha(
    void *q_ptr,
    void *k_ptr,
    void *v_ptr,
    void *o_ptr,
    void *o_tmp_ptr,
    void *softmax_lse_ptr,

    int32_t *cu_seqlens_q_ptr,
    int32_t *cu_seqlens_k_ptr,

    uint32_t q_row_stride,
    uint32_t k_row_stride,
    uint32_t v_row_stride,
    uint32_t o_row_stride,
    uint32_t o_tmp_row_stride,

    uint32_t q_head_stride,
    uint32_t k_head_stride,
    uint32_t v_head_stride,
    uint32_t o_head_stride,
    uint32_t o_tmp_head_stride,

    uint32_t b,
    uint32_t h,
    uint32_t d,
    float softmax_scale,

    uint32_t seqlen_q,
    uint32_t seqlen_k,

    int is_causal,
    int is_bf16,

    int32_t multi_processor_count,
    int32_t num_splits
) {
    Data_type data_type = !is_bf16 ? DATA_TYPE_FP16 : DATA_TYPE_BF16;

    Launch_params<FMHA_fprop_params> launch_params;

    launch_params.elts_per_thread = 0;
    launch_params.multi_processor_count = multi_processor_count;
    launch_params.stream = 0;
    launch_params.is_dropout = false;
    launch_params.return_softmax = false;

    FMHA_fprop_params &params = launch_params.params;

    // Set the pointers and strides.
    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = o_ptr;
    params.o_tmp_ptr = o_tmp_ptr;

    params.softmax_lse_ptr = softmax_lse_ptr;

    // All stride are in elements, not bytes.

    params.q_row_stride_in_elts = q_row_stride;
    params.k_row_stride_in_elts = k_row_stride;
    params.v_row_stride_in_elts = v_row_stride;
    params.o_row_stride_in_elts = o_row_stride;
    params.o_tmp_row_stride_in_elts = o_tmp_row_stride;

    params.q_head_stride_in_elts = q_head_stride;
    params.k_head_stride_in_elts = k_head_stride;
    params.v_head_stride_in_elts = v_head_stride;
    params.o_head_stride_in_elts = o_head_stride;
    params.o_tmp_head_stride_in_elts = o_tmp_head_stride;

    // Set the dimensions.
    params.h = h;
    params.b = b;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.d = d;

    // Set the different scale values.
    const float scale_bmm1 = softmax_scale;
    params.scale_bmm1f = scale_bmm1;
    set_alpha(params.scale_bmm1, scale_bmm1, data_type);

    params.p_dropout = 1.; // probability to keep
    params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_bmm1_rp_dropout = params.rp_dropout * params.scale_bmm1f;
    set_alpha(params.scale_dropout, params.rp_dropout, data_type);

    params.is_bf16 = is_bf16;
    params.is_causal = is_causal;

    params.cu_seqlens_q = cu_seqlens_q_ptr;
    params.cu_seqlens_k = cu_seqlens_k_ptr;

    params.num_splits = num_splits;

    run_fmha_fwd(launch_params);
}
