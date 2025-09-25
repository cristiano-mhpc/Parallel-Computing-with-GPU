#include <iostream>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

template <typename T>
class Cc_dev_ptr {
    static void freecuda(void* ptr) { cudaFree(ptr); }
    std::unique_ptr<T[], decltype(&freecuda)> elem;
public:
    T* get() const { return elem.get(); }
    Cc_dev_ptr(const std::size_t N) : elem{nullptr, freecuda} {
        T* tmp;
        cudaMalloc(reinterpret_cast<void**>(&tmp), N * sizeof(T));
        elem.reset(tmp);
    }
};

__global__ void fillIndices(int* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = idx;
    }
}

int main() {
    const int N = 10;

    Cc_dev_ptr<int> device_data(N);
    std::vector<int> host_data(N);

    fillIndices<<<(N + 255) / 256, 256>>>(device_data.get(), N);
    cudaMemcpy(host_data.data(), device_data.get(), N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        std::cout << "host_data[" << i << "] = " << host_data[i] << std::endl;
    }

    // No need to call cudaFree; Cc_dev_ptr will automatically release the memory.

    return 0;
}
 
