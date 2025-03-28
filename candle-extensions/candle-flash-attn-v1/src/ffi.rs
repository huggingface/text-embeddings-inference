use core::ffi::{c_int, c_void};

extern "C" {
    pub(crate) fn run_mha(
        q_ptr: *const c_void,
        k_ptr: *const c_void,
        v_ptr: *const c_void,
        o_ptr: *const c_void,
        o_tmp_ptr: *const c_void,
        softmax_lse_ptr: *const c_void,
        cu_seqlens_q_ptr: *const i32,
        cu_seqlens_k_ptr: *const i32,

        q_row_stride: u32,
        k_row_stride: u32,
        v_row_stride: u32,
        o_row_stride: u32,
        o_tmp_row_stride: u32,

        q_head_stride: u32,
        k_head_stride: u32,
        v_head_stride: u32,
        o_head_stride: u32,
        o_tmp_head_stride: u32,

        b: u32,
        h: u32,
        d: u32,
        softmax_scale: f32,

        seqlen_q: u32,
        seqlen_k: u32,

        is_causal: c_int,
        is_bf16: c_int,

        multi_processor_count: i32,
        num_splits: i32,
    );

}
