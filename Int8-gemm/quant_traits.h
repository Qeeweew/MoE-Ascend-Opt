#pragma once
#include <cstdint>
#include <torch/extension.h>

namespace quant {

enum class QuantType {
    Q8_0 = 0,  // Default for backward compatibility
    Q4_0 = 1
};

// Base quantization traits template
template <QuantType QT>
struct QuantTraits;

// Q8_0 specialization: int8 quantization with group size 32
template <>
struct QuantTraits<QuantType::Q8_0> {
    static constexpr int QK = 32;           // Elements per quantization block
    static constexpr int MR = 4;            // Microkernel rows
    static constexpr int NR = 8;            // Microkernel cols
    using storage_type = int8_t;            // Quantized storage type
    using scale_type = at::Half;            // Scale/dequantization type
    static constexpr int bytes_per_element = 1;
    static constexpr const char* name = "Q8_0";

    // For compatibility with existing code
    static constexpr int QK8_0 = QK;
};

// Q4_0 specialization: 4-bit quantization with group size 32
// Input is already packed as uint32 (8 x 4-bit elements per uint32)
template <>
struct QuantTraits<QuantType::Q4_0> {
    static constexpr int QK = 32;           // Elements per quantization block
    static constexpr int MR = 4;            // Microkernel rows
    static constexpr int NR = 8;            // Microkernel cols
    using storage_type = uint32_t;          // Packed 4-bit storage (8 elements per uint32)
    using scale_type = at::Half;            // Scale/dequantization type
    static constexpr int bytes_per_element = 4;  // sizeof(uint32_t)
    static constexpr int elements_per_storage = 8;  // 4-bit elements per uint32
    static constexpr const char* name = "Q4_0";

    // For compatibility
    static constexpr int QK4_0 = QK;
};

} // namespace quant

// Python-friendly enum values
namespace {
    constexpr int QUANT_TYPE_Q8_0 = static_cast<int>(quant::QuantType::Q8_0);
    constexpr int QUANT_TYPE_Q4_0 = static_cast<int>(quant::QuantType::Q4_0);
}
