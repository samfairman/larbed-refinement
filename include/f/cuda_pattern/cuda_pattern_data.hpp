#ifndef XEOTSSNSLJPPAXIQVHFMMKPIINJRBEISHMGETNKNWBDFWXJDSTBBSVNEWCNNXAIWLOMTBKGRO
#define XEOTSSNSLJPPAXIQVHFMMKPIINJRBEISHMGETNKNWBDFWXJDSTBBSVNEWCNNXAIWLOMTBKGRO

#include <f/host/cuda/cuda.hpp>
#include <f/cuda_pattern/cuda_pattern_config.hpp>
#include <f/pattern/pattern.hpp>

#include <iostream>

namespace f
{

    struct cuda_pattern_data
    {

        typedef unsigned long int           size_type;
        typedef double                      value_type;
        typedef double2                     complex_type;

        int                                 device_id;
        size_type*                          ar;
        size_type*                          dim;
        value_type*                         I_diff;
        value_type*                         I_exp;
        value_type*                         I_zigmoid;
        value_type*                         diag;
        value_type*                         ug;
        complex_type*                       cache;

        value_type*                         beams;
        value_type*                         kt_factor;

        value_type*                         weights;

        cuda_pattern_data( cuda_pattern_config const& cpc )
        {
            device_id = cpc.device_id;

            int current_id;
            cuda_assert( cudaGetDevice(&current_id) );
            if ( current_id != device_id ) cuda_assert( cudaSetDevice( device_id ) );

            size_type const kt_factor_size = sizeof(value_type) * cpc.tilt_size * 3;
            cuda_assert( cudaMalloc( reinterpret_cast<void**>(&kt_factor), kt_factor_size ) );
            cuda_assert( cudaMemset( reinterpret_cast<void*>(kt_factor), 0, kt_factor_size ) );

            size_type const beams_size = sizeof(value_type) * cpc.tilt_size * 10;
            cuda_assert( cudaMalloc( reinterpret_cast<void**>(&beams), beams_size ) );
            cuda_assert( cudaMemset( reinterpret_cast<void*>(beams), 0, beams_size ) );

            size_type const ug_size = sizeof(value_type) * cpc.ug_size * 2;
            cuda_assert( cudaMalloc( reinterpret_cast<void**>(&ug), ug_size ) );
            cuda_assert( cudaMemset( reinterpret_cast<void*>(ug), 0, ug_size ) );

            size_type const ar_size = sizeof(size_type) * cpc.tilt_size * cpc.max_dim * cpc.max_dim;
            cuda_assert( cudaMalloc( reinterpret_cast<void**>(&ar), ar_size ) );
            cuda_assert( cudaMemset( reinterpret_cast<void*>(ar), 0, ar_size ) );

            size_type const diag_size = sizeof(value_type) * cpc.tilt_size * cpc.max_dim;
            cuda_assert( cudaMalloc( reinterpret_cast<void**>(&diag), diag_size ) );
            cuda_assert( cudaMemset( reinterpret_cast<void*>(diag), 0, diag_size ) );

            size_type const dim_size = sizeof(size_type) * cpc.tilt_size;
            cuda_assert( cudaMalloc( reinterpret_cast<void**>(&dim), dim_size ) );
            cuda_assert( cudaMemset( reinterpret_cast<void*>(dim), 0, dim_size ) );

            size_type const I_exp_size = sizeof(value_type) * cpc.tilt_size * cpc.max_dim;
            cuda_assert( cudaMalloc( reinterpret_cast<void**>(&I_exp), I_exp_size ) );
            cuda_assert( cudaMemset( reinterpret_cast<void*>(I_exp), 0, I_exp_size ) );

            size_type const I_diff_size = sizeof(value_type) * cpc.tilt_size * cpc.max_dim;
            cuda_assert( cudaMalloc( reinterpret_cast<void**>(&I_diff), I_diff_size ) );
            cuda_assert( cudaMemset( reinterpret_cast<void*>(I_diff), 0, I_diff_size ) );

            size_type const I_zigmoid_size = sizeof(value_type) * cpc.tilt_size * cpc.max_dim;
            cuda_assert( cudaMalloc( reinterpret_cast<void**>(&I_zigmoid), I_zigmoid_size ) );
            cuda_assert( cudaMemset( reinterpret_cast<void*>(I_zigmoid), 0, I_zigmoid_size ) );

            size_type const weights_size = sizeof(value_type) * cpc.tilt_size * cpc.max_dim;
            cuda_assert( cudaMalloc( reinterpret_cast<void**>(&weights), weights_size ) );
            cuda_assert( cudaMemset( reinterpret_cast<void*>(weights), 0, weights_size ) );

            size_type const cache_size = sizeof(complex_type) * cpc.tilt_size * cpc.max_dim * cpc.max_dim * 6;
            cuda_assert( cudaMalloc( reinterpret_cast<void**>(&cache), cache_size ) );
            cuda_assert( cudaMemset( reinterpret_cast<void*>(cache), 0, cache_size ) );
        }

        ~cuda_pattern_data()
        {
            int current_id;
            cuda_assert( cudaGetDevice(&current_id) );
            if ( current_id != device_id ) cuda_assert( cudaSetDevice( device_id ) );

            if ( ar ) cuda_assert( cudaFree(ar) );
            if ( dim ) cuda_assert( cudaFree(dim) );
            if ( I_diff ) cuda_assert( cudaFree(I_diff) );
            if ( I_exp ) cuda_assert( cudaFree(I_exp) );
            if ( I_exp ) cuda_assert( cudaFree(I_zigmoid) );
            if ( I_exp ) cuda_assert( cudaFree(weights) );
            if ( diag ) cuda_assert( cudaFree(diag) );
            if ( ug ) cuda_assert( cudaFree(ug) );
            if ( cache ) cuda_assert( cudaFree(cache) );
            if ( beams ) cuda_assert( cudaFree(beams) );
            if ( kt_factor ) cuda_assert( cudaFree(kt_factor) );

            ar = nullptr;
            dim = nullptr;
            I_diff = nullptr;
            I_exp = nullptr;
            I_zigmoid = nullptr;
            weights = nullptr;
            diag = nullptr;
            ug = nullptr;
            cache = nullptr;
        }

    };//struct cuda_pattern_data

}//namespace f

#endif//XEOTSSNSLJPPAXIQVHFMMKPIINJRBEISHMGETNKNWBDFWXJDSTBBSVNEWCNNXAIWLOMTBKGRO

