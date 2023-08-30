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
// UNSUPPORTED: pre-sm-70

// <cuda/barrier>

#include "arrive_tx.h"

int main(int, char**)
{
    BlockSize block_size = BlockSize::CTA;

    NV_IF_ELSE_TARGET(NV_IS_HOST, (
        //Required by concurrent_agents_launch to know how many we're launching
        cuda_thread_count = block_size;

        test<cuda::barrier<cuda::thread_scope_block>, local_memory_selector>(block_size);
    ),(
        test<cuda::barrier<cuda::thread_scope_block>, shared_memory_selector>(block_size);
        test<cuda::barrier<cuda::thread_scope_block>, global_memory_selector>(block_size);
    ));
    return 0;
}
