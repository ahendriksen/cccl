//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: pre-sm-90

// <cuda/ptx>

#include <cuda/ptx>

#include <cuda/std/utility>

#include "concurrent_agents.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

int main(int, char**)
{
    NV_IF_TARGET(NV_IS_DEVICE, (
        // Do not execute. Just check if below PTX compiles (that is: assembles) without error.
        if (false) {
            using cuda::ptx::sem_release;
            using cuda::ptx::sem_acquire;
            using cuda::ptx::sem_weak;
            using cuda::ptx::space_shared_cluster;
            using cuda::ptx::space_shared;
            using cuda::ptx::scope_cluster;
            using cuda::ptx::scope_cta;

            __shared__ uint64_t bar;
            cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cta, space_shared, &bar, 1);
            cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_shared, &bar, 1);

            cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cta, space_shared_cluster, &bar, 1);
            cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_shared_cluster, &bar, 1);

            cuda::ptx::mbarrier_try_wait(sem_acquire, scope_cta, space_shared, &bar, 1);
            cuda::ptx::mbarrier_try_wait(sem_acquire, scope_cluster, space_shared, &bar, 1);
            cuda::ptx::mbarrier_try_wait(sem_acquire, scope_cta, space_shared, &bar, 1, 1);
            cuda::ptx::mbarrier_try_wait(sem_acquire, scope_cluster, space_shared, &bar, 1, 1);

            cuda::ptx::mbarrier_try_wait_parity(sem_acquire, scope_cta, space_shared, &bar, false);
            cuda::ptx::mbarrier_try_wait_parity(sem_acquire, scope_cluster, space_shared, &bar, false);
            cuda::ptx::mbarrier_try_wait_parity(sem_acquire, scope_cta, space_shared, &bar, false, 1);
            cuda::ptx::mbarrier_try_wait_parity(sem_acquire, scope_cluster, space_shared, &bar, false, 1);

            // st.async b32
            cuda::ptx::st_async(sem_weak, space_shared_cluster, nullptr, 1, &bar);
            cuda::ptx::st_async(sem_weak, space_shared_cluster, nullptr, {1}, &bar);
            cuda::ptx::st_async(sem_weak, space_shared_cluster, nullptr, {1, 2}, &bar);
            cuda::ptx::st_async(sem_weak, space_shared_cluster, nullptr, {1, 2, 3, 4}, &bar);

            // st.async b64
            cuda::ptx::st_async(sem_weak, space_shared_cluster, nullptr, 1ul, &bar);
            cuda::ptx::st_async(sem_weak, space_shared_cluster, nullptr, {1ul}, &bar);
            cuda::ptx::st_async(sem_weak, space_shared_cluster, nullptr, {1ul, 2ul}, &bar);
            // The below variant does not exist: (v4.b64)
            // cuda::ptx::st_async(sem_weak, space_shared_cluster, nullptr, {1ul, 2ul, 3ul, 4ul}, &bar);

            // st.async f32
            cuda::ptx::st_async(sem_weak, space_shared_cluster, nullptr, 1.f, &bar);
            cuda::ptx::st_async(sem_weak, space_shared_cluster, nullptr, {1.f}, &bar);
            cuda::ptx::st_async(sem_weak, space_shared_cluster, nullptr, {1.f, 2.f}, &bar);
            cuda::ptx::st_async(sem_weak, space_shared_cluster, nullptr, {1.f, 2.f, 3.f, 4.f}, &bar);

            // st.async f64
            cuda::ptx::st_async(sem_weak, space_shared_cluster, nullptr, 1., &bar);
            cuda::ptx::st_async(sem_weak, space_shared_cluster, nullptr, {1.}, &bar);
            cuda::ptx::st_async(sem_weak, space_shared_cluster, nullptr, {1., 2.}, &bar);
            // The below variant does not exist: (v4.b64)
            // cuda::ptx::st_async(sem_weak, space_shared_cluster, nullptr, {1., 2., 3., 4.}, &bar);

        }
    ));

    return 0;
}
