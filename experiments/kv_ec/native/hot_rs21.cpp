// Opt-in RS(2,1), polynomial 0x11d. No PyTorch dependency; CPU oracle in Python.
#include <cstddef>
#include <cstdint>
#include <cstring>

static inline uint64_t twice(uint64_t v) {
  return ((v & UINT64_C(0x7f7f7f7f7f7f7f7f)) << 1) ^
         (((v >> 7) & UINT64_C(0x0101010101010101)) * 0x1d);
}
static inline uint64_t half(uint64_t v) {
  return ((v >> 1) & UINT64_C(0x7f7f7f7f7f7f7f7f)) ^
         ((v & UINT64_C(0x0101010101010101)) * 0x8e);
}

// op 0/1: a XOR 2*b; op 2: (a XOR b)/2; op 3: explicit XOR-only codec.
extern "C" int hot_rs21_cpu(const uint8_t* a, const uint8_t* b,
                            uint8_t* out, size_t n, int op) {
  if (!a || !b || !out || !n || op < 0 || op > 3) return -1;
  size_t i = 0;
  for (; n - i >= sizeof(uint64_t); i += sizeof(uint64_t)) {
    uint64_t x, y, z;
    // memcpy is safe for unaligned tails/views and compiles to vector loads.
    std::memcpy(&x, a + i, sizeof(x));
    std::memcpy(&y, b + i, sizeof(y));
    z = op == 2 ? half(x ^ y) : x ^ (op == 3 ? y : twice(y));
    std::memcpy(out + i, &z, sizeof(z));
  }
  for (; i < n; ++i) {
    unsigned x = a[i], y = b[i];
    unsigned z = x ^ y;
    out[i] = op == 2 ? (z >> 1) ^ ((z & 1) * 0x8e)
                     : x ^ (op == 3 ? y : ((y << 1) & 255) ^ ((y >> 7) * 0x1d));
  }
  return 0;
}
