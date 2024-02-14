
#include<cuda/atomic>

template <typename T, typename V> union U { T t; V v; };
using atom_t = cuda::atomic<int, cuda::thread_scope_device>*;
using aref_t = cuda::atomic_ref<int, cuda::thread_scope_device>;

// Type your code here, or load an example.
__global__ void square(int* data,
                       int* array,
                       int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        asm volatile("// Before atom_{ref} definition" ::: "memory");
        // Spill to  local happens here (for atomic_ref). (st.local)
        #ifdef AREF
        auto ref = aref_t{*(data + tid)};
        #else
        auto& ref = *U<atom_t, aref_t>{ .v = aref_t{*(data + tid)} }.t;
        #endif
        asm volatile("// After atom_{ref} definition" ::: "memory");

        ref.compare_exchange_strong(array[tid], tid, cuda::std::memory_order_acquire);
    }
}
