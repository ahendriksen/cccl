// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MEMORY_VOIDIFY_H
#define _LIBCUDACXX___MEMORY_VOIDIFY_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__memory/addressof.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <typename _Tp>
_LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX20 void* __voidify(_Tp& __from) {
  // Cast away cv-qualifiers to allow modifying elements of a range through const iterators.
  return const_cast<void*>(static_cast<const volatile void*>(_CUDA_VSTD::addressof(__from)));
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___MEMORY_VOIDIFY_H
