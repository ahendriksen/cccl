//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: true

// test set_terminate

#include <cuda/std/__exception>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ void f1() {}
__host__ __device__ void f2() {}

int main(int, char**)
{
    cuda::std::set_terminate(f1);
    assert(cuda::std::set_terminate(f2) == f1);

  return 0;
}
