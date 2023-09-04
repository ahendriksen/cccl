//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: pre-sm-90

// <cuda/barrier>

#include <cuda/barrier>

#if defined(__CUDA_MINIMUM_ARCH__) && 900 <= __CUDA_MINIMUM_ARCH__
# ifndef __cccl_lib_experimental_ctk12_cp_async_exposure
static_assert(false, "should define __cccl_lib_experimental_ctk12_cp_async_exposure");
# endif
#endif // __CUDA_MINIMUM_ARCH__

int main(int, char**){
    return 0;
}
