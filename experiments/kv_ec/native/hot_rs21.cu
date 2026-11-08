// Vectorized GPU encode/decode. Optional P2P is GPU peer memory, NOT GPUDirect.
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

__device__ inline uint64_t transform(uint64_t x, uint64_t y, int op) {
  if (op == 3) return x ^ y;
  if (op == 2) {
    uint64_t z = x ^ y;
    return ((z >> 1) & 0x7f7f7f7f7f7f7f7fULL) ^ ((z & 0x0101010101010101ULL) * 0x8e);
  }
  return x ^ (((y & 0x7f7f7f7f7f7f7f7fULL) << 1) ^
               (((y >> 7) & 0x0101010101010101ULL) * 0x1d));
}

__global__ void rs_words(const uint64_t* a, const uint64_t* b, uint64_t* out,
                         size_t words, int op) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
       i += size_t(blockDim.x) * gridDim.x) out[i] = transform(a[i], b[i], op);
}

__global__ void rs_bytes(const uint8_t* a, const uint8_t* b, uint8_t* out,
                         size_t start, size_t n, int op) {
  for (size_t i = start + blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += size_t(blockDim.x) * gridDim.x) {
    unsigned x = a[i], y = b[i], z = x ^ y;
    out[i] = op == 2 ? (z >> 1) ^ ((z & 1) * 0x8e)
                     : x ^ (op == 3 ? y : ((y << 1) & 255) ^ ((y >> 7) * 0x1d));
  }
}

extern "C" int hot_rs21_enable_peer(int target, int source) {
  int current = -1, can = 0;
  cudaError_t e = cudaGetDevice(&current);
  if (e != cudaSuccess) return int(e);
  if (current != target || target == source) return int(cudaErrorInvalidDevice);
  e = cudaDeviceCanAccessPeer(&can, target, source);
  if (e != cudaSuccess) return int(e);
  if (!can) return int(cudaErrorPeerAccessUnsupported);
  e = cudaDeviceEnablePeerAccess(source, 0);
  if (e == cudaErrorPeerAccessAlreadyEnabled) {
    cudaGetLastError();
    return 0;
  }
  return int(e);
}

extern "C" int hot_rs21_cuda(const uint8_t* a, const uint8_t* b, uint8_t* out,
                             size_t n, int op, void* stream_ptr) {
  if (!a || !b || !out || !n || op < 0 || op > 3) return int(cudaErrorInvalidValue);
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  size_t start = 0;
  if (((uintptr_t(a) | uintptr_t(b) | uintptr_t(out)) & 7) == 0 && n >= 8) {
    size_t words = n / 8;
    unsigned blocks = unsigned((words + 255) / 256);
    if (blocks > 4096) blocks = 4096;
    rs_words<<<blocks, 256, 0, stream>>>(reinterpret_cast<const uint64_t*>(a),
        reinterpret_cast<const uint64_t*>(b), reinterpret_cast<uint64_t*>(out), words, op);
    start = words * 8;
  }
  if (start < n) {
    size_t blocks = (n - start + 255) / 256;
    if (blocks > 4096) blocks = 4096;
    rs_bytes<<<unsigned(blocks), 256, 0, stream>>>(a, b, out, start, n, op);
  }
  return int(cudaGetLastError());
}
