#pragma once

#include <unordered_map>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>


//#ifdef OLD_GENERATOR_PATH
//#include <ATen/CUDAGeneratorImpl.h>
//#else
//#include <ATen/cuda/CUDAGeneratorImpl.h>
//#endif

namespace layer_norm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Params>
struct LaunchParams{

    size_t elts_per_thread;
    size_t workspace_bytes;
    size_t barrier_size;

    int multi_processor_count;

    cudaStream_t stream;

    Params params;

};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct ParamsBase {
    ParamsBase()
        : ctas_per_col(0)
        , rows(0)
        , cols(0)
        , x(nullptr)
        , mu(nullptr)
        , rs(nullptr)
        , gamma(nullptr)
        , gamma1(nullptr)
        , rowscale(nullptr)
        , colscale(nullptr)
        , dropout_keep_p(1.f)
        , dropout_scale(1.f)
        , is_rms_norm(false)
        , workspace(nullptr)
        , barrier(nullptr)
    {
    }

    // For Multi-CTA, number of different CTA groups. Otherwise same as gridDim.x.
    int ctas_per_col;

    // Input is interpreted as matrix. We normalize across columns.
    int rows;
    int cols;

    // Common data pointers.
    void *x0;
    void *x1;
    void *residual;
    void *x;
    void *dmask;
    void *dmask1;
    void *mu;
    void *rs;
    void *gamma;
    void *gamma1;
    void *rowscale;
    void *colscale;
    void *x0_subset;
    void *z_subset;

    float inverse_cols;

    float dropout_keep_p;
    float dropout_scale;
    float rowscale_const;

    bool is_rms_norm;

    // Multi-CTA workspace in gmem.
    void *workspace;

    // Multi-CTA sync barriers in gmem.
    int *barrier;

};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct FwdParams : public ParamsBase {
    FwdParams()
        : ParamsBase()
        , z(nullptr)
        , z1(nullptr)
        , beta(nullptr)
        , beta1(nullptr)
        , epsilon(0.f)
    {
    }

    // Output of LN FWD.
    void *z;
    void *z1;
    void *beta;
    void *beta1;
    float epsilon;

    // Random state.
    // at::PhiloxCudaState philox_args;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using FwdFunction = std::function<void(LaunchParams<FwdParams>&, const bool)>;
using FunctionKey = uint64_t;
using FwdRegistry = std::unordered_map<FunctionKey, FwdFunction>;

extern FwdRegistry FWD_FUNCS;

////////////////////////////////////////////////////////////////////////////////////////////////////

using fp32 = float;
using fp16 = half;
using bf16 = nv_bfloat16;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct TypeId{};

template<>
struct TypeId<fp16>{
    constexpr static uint32_t Value = 0;
};

template<>
struct TypeId<bf16>{
    constexpr static uint32_t Value = 1;
};

template<>
struct TypeId<fp32>{
    constexpr static uint32_t Value = 2;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int S>
struct Type2Key{
    constexpr static uint32_t Value = TypeId<T>::Value << S;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct WeightType2Key : public Type2Key<T, 0>{};

template<typename T>
struct InputType2Key : public Type2Key<T, 2>{};

template<typename T>
struct ResidualType2Key : public Type2Key<T, 4>{};

template<typename T>
struct OutputType2Key : public Type2Key<T, 6>{};

template<typename T>
struct ComputeType2Key : public Type2Key<T, 8>{};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename W, typename I, typename R, typename O, typename C>
struct Types2Key{
    constexpr static uint32_t Value = WeightType2Key<W>::Value | InputType2Key<I>::Value | ResidualType2Key<R>::Value | OutputType2Key<O>::Value | ComputeType2Key<C>::Value;
    constexpr static inline uint64_t get(const uint64_t hidden_size){
        constexpr uint64_t type_key = Value;
        return (type_key << 32) | hidden_size;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename W, typename I, typename R, typename O, typename C, uint64_t HIDDEN_SIZE>
struct FwdRegistrar{
    FwdRegistrar(FwdFunction f){
        uint64_t key = Types2Key<W,I,R,O,C>::get(HIDDEN_SIZE);
        FWD_FUNCS.insert({ key, f });
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace layer_norm
