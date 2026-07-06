#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include <type_traits>

using namespace AscendC;

// -----------------------------------------------------------------------------
// Type Traits for BFloat16 detection
// -----------------------------------------------------------------------------
template<typename T>
struct IsBFloat16 : std::false_type {};

template<>
struct IsBFloat16<bfloat16_t> : std::true_type {};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
constexpr int32_t BLOCK_SIZE       = 128; // 一次处理的计算块大小
constexpr int32_t GROUP_SIZE       = 32;  // 现在的量化 group 大小
constexpr int32_t GROUPS_PER_BLOCK = BLOCK_SIZE / GROUP_SIZE; // 4个
constexpr int32_t COMPUTE_ROWS     = 32;  // 每次内循环拷贝计算的行数
constexpr int32_t PACK_RATIO       = 8;
constexpr int32_t GROUP_TILE       = 8;

// -----------------------------------------------------------------------------
// HardEvent IDs (only 0/1/2/3)
// Same HardEvent must use the same event_id everywhere.
// -----------------------------------------------------------------------------
static constexpr int EID_MTE2_V = 0;  // MTE2 -> V
static constexpr int EID_V_MTE2 = 1;  // V -> MTE2
static constexpr int EID_V_MTE3 = 2;  // V -> MTE3
static constexpr int EID_MTE3_V = 3;  // MTE3 -> V

// -----------------------------------------------------------------------------
// Simple Low-level ZeroOut (no pipe)
// -----------------------------------------------------------------------------
template<typename T>
class KernelZeroOutLL {
public:
    __aicore__ inline void Init(GM_ADDR dst, uint64_t elem_cnt) {
        dstGm.SetGlobalBuffer((__gm__ T*)dst);
        total = elem_cnt;
    }

    __aicore__ inline void Process() {
        uint32_t addr = 0;
        LocalTensor<T> tile = LocalTensor<T>(TPosition::VECOUT, addr, TILE_LEN);

        int32_t core_idx = GetBlockIdx();
        int32_t core_num = GetBlockNum();

        Duplicate(tile, (T)0, TILE_LEN);
        // V -> MTE3
        SetFlag<HardEvent::V_MTE3>(EID_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EID_V_MTE3);

        for (uint64_t base = (uint64_t)core_idx * TILE_LEN; base < total; base += (uint64_t)core_num * TILE_LEN) {
            uint32_t len = (uint32_t)((total - base < (uint64_t)TILE_LEN) ? (total - base) : TILE_LEN);
            DataCopy(dstGm[base], tile, len);
        }
        DataSyncBarrier<MemDsbT::DDR>();
    }

private:
    GlobalTensor<T> dstGm;
    uint64_t total = 0;
    static constexpr int32_t TILE_LEN = 2048;
};

// -----------------------------------------------------------------------------
// Kernel SwiGLU (Low-level, NO double-buffer; simple)
// Layout: [Gate | Value] split at inter_dim
// Input: FP32, Output: T
// -----------------------------------------------------------------------------
template<typename T>
class KernelSwiGLU {
public:
    __aicore__ inline KernelSwiGLU() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, int32_t total_rows, int32_t inter_dim) {
        inputGm.SetGlobalBuffer((__gm__ float*)input);
        outputGm.SetGlobalBuffer((__gm__ T*)output);
        this->total_rows = total_rows;
        this->inter_dim  = inter_dim;

        uint32_t addr = 0;
        f_in  = LocalTensor<float>(TPosition::VECCALC, addr, 2 * TILE_LEN);
        addr += (uint32_t)(2 * TILE_LEN * sizeof(float));
        f_out = LocalTensor<float>(TPosition::VECCALC, addr, TILE_LEN);
        addr += (uint32_t)(TILE_LEN * sizeof(float));

        t_out = LocalTensor<T>(TPosition::VECOUT, addr, TILE_LEN);
        addr += (uint32_t)(TILE_LEN * sizeof(T));
    }

    __aicore__ inline void Process() {
        int32_t core_idx = GetBlockIdx();
        int32_t core_num = GetBlockNum();

        for (int32_t r = core_idx; r < total_rows; r += core_num) {
            ComputeRow(r);
        }
    }

private:
    __aicore__ inline void ComputeRow(int32_t row_idx) {
        uint64_t in_base  = (uint64_t)row_idx * 2u * (uint64_t)inter_dim;
        uint64_t out_base = (uint64_t)row_idx * (uint64_t)inter_dim;

        bool first_tile = true;
        for (int32_t i = 0; i < inter_dim; i += (int32_t)TILE_LEN) {
            int32_t len = (inter_dim - i < (int32_t)TILE_LEN) ? (inter_dim - i) : (int32_t)TILE_LEN;

            // reverse deps to avoid overwrite
            if (!first_tile) {
                WaitFlag<HardEvent::V_MTE2>(EID_V_MTE2);     // previous compute done before overwriting t_in
                WaitFlag<HardEvent::MTE3_V>(EID_MTE3_V);     // previous store done before overwriting t_out
            }

            // Copy FP32 input directly
            DataCopyParams params;
            params.blockCount = 2;
            params.blockLen   = (uint16_t)(len * (int32_t)sizeof(float) / 32);
            params.srcStride  = (uint16_t)((inter_dim - len) * (int32_t)sizeof(float) / 32);
            params.dstStride  = 0;

            DataCopy(f_in, inputGm[in_base + (uint64_t)i], params);
            SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(EID_MTE2_V);

            Silu(f_out, f_in, len);
            Mul(f_out, f_out, f_in[len], len);
            Cast(t_out, f_out, RoundMode::CAST_ROUND, len);

            // mark V done for next tile's CopyIn overwrite protection
            SetFlag<HardEvent::V_MTE2>(EID_V_MTE2);

            // store
            SetFlag<HardEvent::V_MTE3>(EID_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(EID_V_MTE3);

            DataCopy(outputGm[out_base + (uint64_t)i], t_out, len);

            SetFlag<HardEvent::MTE3_V>(EID_MTE3_V);

            first_tile = false;
        }

        // close last outstanding pairs (important across rows, avoid "Set twice" later)
        if (!first_tile) {
            WaitFlag<HardEvent::V_MTE2>(EID_V_MTE2);
            WaitFlag<HardEvent::MTE3_V>(EID_MTE3_V);
        }
    }

private:
    GlobalTensor<float> inputGm;
    GlobalTensor<T> outputGm;

    LocalTensor<float> f_in, f_out;
    LocalTensor<T>     t_out;

    int32_t total_rows = 0;
    int32_t inter_dim  = 0;
    static constexpr int TILE_LEN = 1024;
};

template<typename T, typename OutputT = T>
class KernelGroupedGemvW4A16Moe {
public:
    __aicore__ inline KernelGroupedGemvW4A16Moe() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR scales,
                                GM_ADDR expert_ids, GM_ADDR y, GM_ADDR topk_weights,
                                int32_t total_tokens, int32_t in_dim, int32_t out_dim,
                                int32_t num_experts, int32_t top_k,
                                bool is_broadcast_x, bool is_weighted_sum) {
        this->total_tokens = total_tokens;
        this->top_k = top_k;
        this->in_dim = in_dim;
        this->out_dim = out_dim;
        this->out_dim_packed = out_dim / PACK_RATIO;
        this->num_experts = num_experts;
        // Block 是数据加载单元，Group 是 Scale 单元
        this->num_blocks = in_dim / BLOCK_SIZE;
        this->total_groups = in_dim / GROUP_SIZE;

        this->is_broadcast_x = is_broadcast_x;
        this->is_weighted_sum = is_weighted_sum;

        xGm.SetGlobalBuffer((__gm__ T*)x);
        weightGm.SetGlobalBuffer((__gm__ int32_t*)weight);
        scalesGm.SetGlobalBuffer((__gm__ T*)scales);
        expertIdsGm.SetGlobalBuffer((__gm__ int32_t*)expert_ids);
        yGm.SetGlobalBuffer((__gm__ OutputT*)y);

        if (is_weighted_sum) {
            topkWeightsGm.SetGlobalBuffer((__gm__ float*)topk_weights);
        }

        BuildLocalTensors();
    }

    __aicore__ inline void Process() {
        int32_t total_tasks = total_tokens * num_blocks;
        int32_t core_idx = GetBlockIdx();
        int32_t core_num = GetBlockNum();

        int32_t base = total_tasks / core_num;
        int32_t rem  = total_tasks % core_num;

        int32_t start_task, task_cnt;
        if (core_idx < rem) {
            task_cnt = base + 1;
            start_task = core_idx * (base + 1);
        } else {
            task_cnt = base;
            start_task = rem * (base + 1) + (core_idx - rem) * base;
        }
        int32_t end_task = start_task + task_cnt;
        if (task_cnt <= 0) return;

        for (int32_t n_start = 0; n_start < out_dim; n_start += TILE_N) {
            int32_t cur_n_len = (out_dim - n_start < TILE_N) ? (out_dim - n_start) : TILE_N;
            Duplicate(y_fp32, 0.0f, cur_n_len);

            int32_t current_row_idx = -1;
            if (start_task < end_task) current_row_idx = start_task / num_blocks;

            for (int32_t task_id = start_task; task_id < end_task; ++task_id) {
                int32_t row_idx = task_id / num_blocks;
                int32_t b_idx   = task_id % num_blocks;

                if (row_idx != current_row_idx) {
                    CopyOut(current_row_idx, n_start, cur_n_len);
                    Duplicate(y_fp32, 0.0f, cur_n_len);
                    current_row_idx = row_idx;
                }

                int32_t expert_id = expertIdsGm.GetValue(row_idx);
                ProcessBlock(b_idx, expert_id, row_idx, n_start, cur_n_len, y_fp32);
            }

            if (current_row_idx != -1) {
                CopyOut(current_row_idx, n_start, cur_n_len);
            }
        }
    }

private:
    __aicore__ inline void BuildLocalTensors() {
        constexpr int32_t tile_n_packed = TILE_N / PACK_RATIO;
        uint32_t addr = 0;

        x_local = LocalTensor<T>(TPosition::VECIN, addr, BLOCK_SIZE);
        addr += (uint32_t)(BLOCK_SIZE * sizeof(T));

        w_local[0] = LocalTensor<int32_t>(TPosition::VECIN, addr, COMPUTE_ROWS * tile_n_packed);
        addr += (uint32_t)(COMPUTE_ROWS * tile_n_packed * sizeof(int32_t));
        w_local[1] = LocalTensor<int32_t>(TPosition::VECIN, addr, COMPUTE_ROWS * tile_n_packed);
        addr += (uint32_t)(COMPUTE_ROWS * tile_n_packed * sizeof(int32_t));

        // 保留 4 个 Scale 行缓存
        s_local = LocalTensor<T>(TPosition::VECIN, addr, GROUPS_PER_BLOCK * TILE_N);
        addr += (uint32_t)(GROUPS_PER_BLOCK * TILE_N * sizeof(T));

        y_fp32 = LocalTensor<float>(TPosition::VECOUT, addr, TILE_N);
        addr += (uint32_t)(TILE_N * sizeof(float));

        // 开辟 4 组计算和的缓存区，用于延迟乘 Scale
        group_acc = LocalTensor<half>(TPosition::VECCALC, addr, GROUPS_PER_BLOCK * TILE_N);
        addr += (uint32_t)(GROUPS_PER_BLOCK * TILE_N * sizeof(half) + 256);

        w_half = LocalTensor<half>(TPosition::VECCALC, addr, GROUP_TILE * TILE_N);
        addr += (uint32_t)(GROUP_TILE * TILE_N * sizeof(half));
    }

    __aicore__ inline void PrefetchW(int32_t expert_id, int32_t b_idx, int32_t k_inner_start,
                                     int32_t n_offset, int32_t cur_n_len, int buf_id) {
        uint64_t w_stride_k = (uint64_t)out_dim_packed;
        int32_t cur_n_packed = cur_n_len / PACK_RATIO;
        int32_t n_offset_packed = n_offset / PACK_RATIO;

        uint64_t w_offset = (uint64_t)expert_id * (uint64_t)in_dim * w_stride_k +
                            (uint64_t)(b_idx * BLOCK_SIZE + k_inner_start) * w_stride_k +
                            (uint64_t)n_offset_packed;

        DataCopyExtParams p{
            (uint16_t)COMPUTE_ROWS,
            (uint32_t)(cur_n_packed * sizeof(int32_t)),
            (uint32_t)((w_stride_k - (uint64_t)cur_n_packed) * sizeof(int32_t)),
            0, 0
        };
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(w_local[buf_id], weightGm[w_offset], p, pad);
    }

    __aicore__ inline void PrefetchS(int32_t expert_id, int32_t b_idx,
                                     int32_t n_offset, int32_t cur_n_len) {
        // 当前Block的基础 group 索引
        uint64_t base_group = (uint64_t)b_idx * GROUPS_PER_BLOCK;
        
        DataCopyExtParams p{
            (uint16_t)GROUPS_PER_BLOCK,
            (uint32_t)(cur_n_len * sizeof(T)),
            (uint32_t)((out_dim - (uint64_t)cur_n_len) * sizeof(T)),
            (uint32_t)((TILE_N - cur_n_len) * sizeof(T) / 32), 0
        };

        DataCopyPadExtParams<T> pad{false, 0, 0, 0};

        // 将属于这个Block里的 4 行 Scale 读出来
        uint64_t sz_offset = (uint64_t)expert_id * (uint64_t)total_groups * (uint64_t)out_dim +
                             base_group * (uint64_t)out_dim +
                             (uint64_t)n_offset;
        DataCopyPad(s_local, scalesGm[sz_offset], p, pad);
    }

    __aicore__ inline void ComputeChunk(LocalTensor<int32_t>& w_i32,
                                        int32_t x_offset_idx,
                                        int32_t tile_n,
                                        int32_t tile_n_packed,
                                        LocalTensor<half>& x_full_half,
                                        LocalTensor<half>& current_group_acc) {
        LocalTensor<uint64_t> x_u64 = x_full_half.template ReinterpretCast<uint64_t>();

        constexpr int STEP = 4;
        half x_val_buf[COMPUTE_ROWS];
        uint64_t* x_val_buf_u64 = (uint64_t*)x_val_buf;
        for (int i = 0; i < COMPUTE_ROWS / STEP; ++i) {
            x_val_buf_u64[i] = x_u64.GetValue(x_offset_idx / STEP + i);
        }

        for (int i = 0; i < COMPUTE_ROWS; i += GROUP_TILE) {
            LocalTensor<int4b_t> w_int4 = w_i32[i * tile_n_packed].template ReinterpretCast<int4b_t>();
            Cast(w_half, w_int4, RoundMode::CAST_NONE, GROUP_TILE * tile_n);
            for (int k = 0; k < GROUP_TILE; ++k) {
                Axpy(current_group_acc, w_half[k * tile_n], x_val_buf[i + k], tile_n);
            }
        }
    }

    __aicore__ inline void ProcessBlock(int32_t b_idx, int32_t expert_id, int32_t row_idx,
                                        int32_t n_offset, int32_t cur_n_len,
                                        LocalTensor<float>& global_acc) {
        uint64_t x_offset;
        if (is_broadcast_x) {
            int32_t batch_idx = row_idx / top_k;
            x_offset = (uint64_t)batch_idx * (uint64_t)in_dim + (uint64_t)b_idx * BLOCK_SIZE;
        } else {
            x_offset = (uint64_t)row_idx * (uint64_t)in_dim + (uint64_t)b_idx * BLOCK_SIZE;
        }

        DataCopy(x_local, xGm[x_offset], BLOCK_SIZE);
        SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EID_MTE2_V);

        // 如果是 BF16，需要转换为 half 供后续计算使用
        // 使用 y_fp32 的前 64 个 float (256 bytes) 作为中转，可以容纳 128 个 float
        // 转换流程：bf16 -> fp32 -> fp16
        if constexpr (IsBFloat16<T>::value) {
            LocalTensor<float> f_tmp = w_half.template ReinterpretCast<float>();
            LocalTensor<half> x_half = x_local.template ReinterpretCast<half>();
            // bf16 -> fp32
            Cast(f_tmp, x_local, RoundMode::CAST_NONE, BLOCK_SIZE);
            // fp32 -> fp16 (原地覆盖 x_local)
            Cast(x_half, f_tmp, RoundMode::CAST_ROUND, BLOCK_SIZE);
        }

        PrefetchW(expert_id, b_idx, 0, n_offset, cur_n_len, 0);
        SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);

        // 初始化 4 个 group 的累加器
        Duplicate(group_acc, (half)0.0f, GROUPS_PER_BLOCK * TILE_N);

        int32_t cur_n_packed = cur_n_len / PACK_RATIO;
        int ping = 0;

        for (int32_t k_inner = 0; k_inner < BLOCK_SIZE; k_inner += COMPUTE_ROWS) {
            WaitFlag<HardEvent::MTE2_V>(EID_MTE2_V);
            if (k_inner != 0) {
                WaitFlag<HardEvent::V_MTE2>(EID_V_MTE2);
            }

            if (k_inner + COMPUTE_ROWS < BLOCK_SIZE) {
                int next_buf = ping ^ 1;
                PrefetchW(expert_id, b_idx, k_inner + COMPUTE_ROWS, n_offset, cur_n_len, next_buf);
                SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);
            } else {
                // 最后一次拷贝 4 个对应的 Scale
                PrefetchS(expert_id, b_idx, n_offset, cur_n_len);
                SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);
            }

            // 计算该 32 行该放入哪个 group_acc
            int step = k_inner / COMPUTE_ROWS;
            LocalTensor<half> current_acc = group_acc[step * TILE_N];

            // x_local 现在已经是 half 类型（如果是 bf16 已经转换过了）
            LocalTensor<half> x_input = x_local.template ReinterpretCast<half>();
            ComputeChunk(w_local[ping], k_inner, cur_n_len, cur_n_packed, x_input, current_acc);

            if (k_inner + COMPUTE_ROWS < BLOCK_SIZE) {
                SetFlag<HardEvent::V_MTE2>(EID_V_MTE2);
            }
            ping ^= 1;
        }

        WaitFlag<HardEvent::MTE2_V>(EID_MTE2_V);

        // 对称量化，省去了 zero-point offset 补偿
        // 遍历 4 个分组，将 4 个 FP16 Group累加器 分别乘上对应的 Scale 后加入全局 Float 累加器中
        if constexpr (IsBFloat16<T>::value) {
            // 对于 BF16，需要复用 w_half 作为 fp32 workspace
            // w_half 大小为 GROUP_TILE * TILE_N = 8 * 2048 个 half
            // 作为 fp32 可以容纳 8192 个 float，足够覆盖 cur_n_len (<= 2048)
            LocalTensor<float> f_acc = w_half.template ReinterpretCast<float>();
            LocalTensor<float> f_scale = f_acc[TILE_N];

            for (int i = 0; i < GROUPS_PER_BLOCK; ++i) {
                LocalTensor<half> current_acc = group_acc[i * TILE_N];
                LocalTensor<T> current_scale = s_local[i * TILE_N];

                // current_acc (half) -> f_acc (float)
                Cast(f_acc, current_acc, RoundMode::CAST_NONE, cur_n_len);
                // current_scale (bfloat16) -> f_scale (float)
                Cast(f_scale, current_scale, RoundMode::CAST_NONE, cur_n_len);
                // f_acc * f_scale -> f_acc
                MulAddDst(global_acc, f_acc, f_scale, cur_n_len);
            }
        } else {
            // FP16 使用原来的 MulAddDst
            for (int i = 0; i < GROUPS_PER_BLOCK; ++i) {
                LocalTensor<half> current_acc = group_acc[i * TILE_N];
                LocalTensor<T> current_scale = s_local[i * TILE_N];
                MulAddDst(global_acc, current_acc, current_scale, cur_n_len);
            }
        }
    }

    __aicore__ inline void CopyOut(int32_t row_idx, int32_t n_offset, int32_t cur_n_len) {
        if (row_idx < 0) return;

        if (is_weighted_sum) {
            float w_val = topkWeightsGm.GetValue(row_idx);
            Muls(y_fp32, y_fp32, w_val, cur_n_len);
        }

        SetFlag<HardEvent::V_MTE3>(EID_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EID_V_MTE3);

        if constexpr (std::is_same_v<OutputT, float>) {
            // FP32 output: copy directly without cast
            SetAtomicAdd<float>();

            if (is_weighted_sum) {
                int32_t batch_idx = row_idx / top_k;
                uint64_t y_offset = (uint64_t)batch_idx * (uint64_t)out_dim + (uint64_t)n_offset;
                DataCopy(yGm[y_offset], y_fp32, cur_n_len);
            } else {
                uint64_t y_offset = (uint64_t)row_idx * (uint64_t)out_dim + (uint64_t)n_offset;
                DataCopy(yGm[y_offset], y_fp32, cur_n_len);
            }

            SetAtomicNone();
        } else {
            // T output: cast from FP32 to T
            LocalTensor<T> y_half = y_fp32.template ReinterpretCast<T>();
            Cast(y_half, y_fp32, RoundMode::CAST_ROUND, cur_n_len);

            SetAtomicAdd<T>();

            if (is_weighted_sum) {
                int32_t batch_idx = row_idx / top_k;
                uint64_t y_offset = (uint64_t)batch_idx * (uint64_t)out_dim + (uint64_t)n_offset;
                DataCopy(yGm[y_offset], y_half, cur_n_len);
            } else {
                uint64_t y_offset = (uint64_t)row_idx * (uint64_t)out_dim + (uint64_t)n_offset;
                DataCopy(yGm[y_offset], y_half, cur_n_len);
            }

            SetAtomicNone();
        }

        SetFlag<HardEvent::MTE3_V>(EID_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(EID_MTE3_V);
    }

private:
    GlobalTensor<float> topkWeightsGm;

    GlobalTensor<T>       xGm;
    GlobalTensor<int32_t> weightGm;
    GlobalTensor<T>       scalesGm;
    GlobalTensor<int32_t> expertIdsGm;
    GlobalTensor<OutputT> yGm;

    // UB
    LocalTensor<T>        x_local;
    LocalTensor<int32_t>  w_local[2];
    LocalTensor<T>        s_local;

    LocalTensor<float>    y_fp32;
    LocalTensor<half>     group_acc;
    LocalTensor<half>     w_half;

    int32_t top_k = 0;
    int32_t in_dim = 0;
    int32_t out_dim = 0;
    int32_t out_dim_packed = 0;
    int32_t num_experts = 0;
    int32_t num_blocks = 0;
    int32_t total_groups = 0;
    int32_t total_tokens = 0;
    
    // Runtime flags
    bool is_broadcast_x = false;
    bool is_weighted_sum = false;

    static constexpr int32_t TILE_N = 2048;
};

// -----------------------------------------------------------------------------
// Kernel Cast FP32 to T (for final output conversion)
// -----------------------------------------------------------------------------
template<typename T>
class KernelCastFP32ToT {
public:
    __aicore__ inline KernelCastFP32ToT() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, int32_t total_elems) {
        inputGm.SetGlobalBuffer((__gm__ float*)input);
        outputGm.SetGlobalBuffer((__gm__ T*)output);
        this->total_elems = total_elems;

        uint32_t addr = 0;
        f_local = LocalTensor<float>(TPosition::VECIN, addr, TILE_LEN);
        addr += (uint32_t)(TILE_LEN * sizeof(float));
        t_local = LocalTensor<T>(TPosition::VECOUT, addr, TILE_LEN);
        addr += (uint32_t)(TILE_LEN * sizeof(T));
    }

    __aicore__ inline void Process() {
        int32_t core_idx = GetBlockIdx();
        int32_t core_num = GetBlockNum();

        for (int64_t base = (int64_t)core_idx * TILE_LEN; base < total_elems; base += (int64_t)core_num * TILE_LEN) {
            int32_t len = (int32_t)((total_elems - base < (int64_t)TILE_LEN) ? (total_elems - base) : TILE_LEN);

            DataCopy(f_local, inputGm[base], len);
            SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(EID_MTE2_V);

            Cast(t_local, f_local, RoundMode::CAST_ROUND, len);

            SetFlag<HardEvent::V_MTE3>(EID_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(EID_V_MTE3);

            DataCopy(outputGm[base], t_local, len);

            SetFlag<HardEvent::MTE3_V>(EID_MTE3_V);
            WaitFlag<HardEvent::MTE3_V>(EID_MTE3_V);
        }
    }

private:
    GlobalTensor<float> inputGm;
    GlobalTensor<T> outputGm;

    LocalTensor<float> f_local;
    LocalTensor<T> t_local;

    int64_t total_elems = 0;
    static constexpr uint32_t TILE_LEN = 2048;
};

// -----------------------------------------------------------------------------
// Fused Kernel Implementation
// -----------------------------------------------------------------------------
template<typename T>
__aicore__ inline void fused_moe_small_bs_impl(
    GM_ADDR x,
    GM_ADDR w13_weight, GM_ADDR w13_scales,
    GM_ADDR w2_weight,  GM_ADDR w2_scales,
    GM_ADDR expert_ids, GM_ADDR topk_weights,
    GM_ADDR y,
    GM_ADDR workspace,
    int32_t batch_size, int32_t hidden_size, int32_t inter_size,
    int32_t num_experts, int32_t top_k)
{
    int32_t total_tokens = batch_size * top_k;

    // workspace layout:
    // [0 : w13_out_size) : W13 output (FP32) [TotalTokens, 2 * InterDim]
    // [w13_out_size : w13_out_size + w2_in_size) : SwiGLU output (T) [TotalTokens, InterDim]
    // [w13_out_size + w2_in_size : end) : W2 output (FP32) [BatchSize, HiddenSize]
    GM_ADDR w13_out_ptr = workspace;
    uint64_t w13_out_size = (uint64_t)total_tokens * (uint64_t)(inter_size * 2) * sizeof(float);

    GM_ADDR w2_in_ptr = (GM_ADDR)((__gm__ uint8_t*)w13_out_ptr + w13_out_size);
    uint64_t w2_in_size = (uint64_t)total_tokens * (uint64_t)inter_size * sizeof(T);

    GM_ADDR w2_out_ptr = (GM_ADDR)((__gm__ uint8_t*)w2_in_ptr + w2_in_size);
    uint64_t w2_out_size = (uint64_t)batch_size * (uint64_t)hidden_size * sizeof(float);

    // Phase 0: Zero out entire workspace at once (W13 FP32 + W2_in T + W2 FP32)
    {
        uint64_t total_workspace_size = w13_out_size + w2_in_size + w2_out_size;
        uint64_t total_elems = total_workspace_size / sizeof(int32_t);

        KernelZeroOutLL<int32_t> zeroOp;
        zeroOp.Init(workspace, total_elems);
        zeroOp.Process();
    }

    AscendC::SyncAll();

    // Phase 1: W13 Gemv (Broadcast X = true, Weighted = false, FP32 output)
    {
        KernelGroupedGemvW4A16Moe<T, float> op_w13;
        op_w13.Init(x, w13_weight, w13_scales,
                    expert_ids, w13_out_ptr, nullptr,
                    total_tokens, hidden_size, inter_size * 2, num_experts, top_k,
                    true, false);
        op_w13.Process();
    }

    AscendC::SyncAll();

    // Phase 2: SwiGLU (FP32 input, T output)
    // SwiGLU reads from w13_out and writes to w2_in (separate memory areas)
    {
        KernelSwiGLU<T> op_act;
        op_act.Init(w13_out_ptr, w2_in_ptr, total_tokens, inter_size);
        op_act.Process();
    }

    AscendC::SyncAll();

    // Phase 3: W2 Gemv (Broadcast X = false, Weighted = true, FP32 output)
    {
        KernelGroupedGemvW4A16Moe<T, float> op_w2;
        op_w2.Init(w2_in_ptr, w2_weight, w2_scales,
                   expert_ids, w2_out_ptr, topk_weights,
                   total_tokens, inter_size, hidden_size, num_experts, top_k,
                   false, true);
        op_w2.Process();
    }

    AscendC::SyncAll();

    // Phase 4: Cast FP32 to T (final output to y)
    {
        KernelCastFP32ToT<T> op_cast;
        op_cast.Init(w2_out_ptr, y, (int32_t)((uint64_t)batch_size * (uint64_t)hidden_size));
        op_cast.Process();
    }
}

// -----------------------------------------------------------------------------
// Extern C Entry Points
// -----------------------------------------------------------------------------
/*
llvm-objdump -t ./device_aiv.o | grep "fused_moe_small_bs_w4a16_bf16_17_mix_aiv"
000000000001d794 l       .text  0000000000000000 fused_moe_small_bs_w4a16_bf16_17_mix_aiv$local
0000000000000000 l     O .ascend.meta.fused_moe_small_bs_w4a16_bf16_17_mix_aiv  0000000000000010 _ZL45fused_moe_small_bs_w4a16_bf16_mix_aiv_section
0000000000000000 l       .ascend.meta.fused_moe_small_bs_w4a16_bf16_17_mix_aiv  0000000000000000 $d.2
0000000000000000 l    d  .ascend.meta.fused_moe_small_bs_w4a16_bf16_17_mix_aiv  0000000000000000 .ascend.meta.fused_moe_small_bs_w4a16_bf16_17_mix_aiv
000000000001d794 g     F .text  0000000000002458 fused_moe_small_bs_w4a16_bf16_17_mix_aiv
*/
extern "C" __global__ __aicore__ void fused_moe_small_bs_w4a16_bf16(
    GM_ADDR x,
    GM_ADDR w13_weight, GM_ADDR w13_scales,
    GM_ADDR w2_weight,  GM_ADDR w2_scales,
    GM_ADDR expert_ids, GM_ADDR topk_weights,
    GM_ADDR y,
    GM_ADDR workspace,
    int32_t batch_size, int32_t hidden_size, int32_t inter_size,
    int32_t num_experts, int32_t top_k)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::InitSocState();
    AscendC::ICachePreLoad(5);

    fused_moe_small_bs_impl<bfloat16_t>(x,
                                        w13_weight, w13_scales,
                                        w2_weight,  w2_scales,
                                        expert_ids, topk_weights,
                                        y, workspace,
                                        batch_size, hidden_size, inter_size,
                                        num_experts, top_k);
}

// -----------------------------------------------------------------------------
// Kernel Batch Gemv (Batch <= 8, Split-K, Group-level Double Buffering)
// Optimized with DataCopyPad for strided memory transfers
// -----------------------------------------------------------------------------
template<typename T, typename OutputT = T, int MAX_BS=4>
class KernelBatchGemmW4A16 {
public:
    __aicore__ inline KernelBatchGemmW4A16() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR scales, GM_ADDR y,
                                int32_t batch_size, int32_t in_dim, int32_t out_dim) {
        this->batch_size = batch_size;
        this->in_dim = in_dim;
        this->out_dim = out_dim;
        this->total_groups = in_dim / GROUP_SIZE;
        this->out_dim_packed = out_dim / PACK_RATIO;
        
        // Dynamic Split-K Calculation
        int32_t core_num = GetBlockNum();
        this->num_n_tiles = (out_dim + TILE_N - 1) / TILE_N;
        this->split_k = core_num / this->num_n_tiles;
        if (this->split_k < 1) {
            this->split_k = 1;
        }
        this->total_tasks = this->num_n_tiles * this->split_k;

        xGm.SetGlobalBuffer((__gm__ T*)x);
        weightGm.SetGlobalBuffer((__gm__ int32_t*)weight);
        scalesGm.SetGlobalBuffer((__gm__ T*)scales);
        yGm.SetGlobalBuffer((__gm__ OutputT*)y);

        BuildLocalTensors();
    }

    __aicore__ inline void Process() {
        int32_t core_idx = GetBlockIdx();
        int32_t core_num = GetBlockNum();

        for (int32_t task_id = core_idx; task_id < total_tasks; task_id += core_num) {
            ProcessTask(task_id);
        }
    }

private:
    __aicore__ inline void BuildLocalTensors() {
        uint32_t addr = 0;

        // X buffer (2 for double buffering). Need to hold BatchSize * GROUP_SIZE
        x_local[0] = LocalTensor<T>(TPosition::VECIN, addr, MAX_BS * GROUP_SIZE); // max_bs=8
        addr += (uint32_t)(MAX_BS * GROUP_SIZE * sizeof(T));
        x_local[1] = LocalTensor<T>(TPosition::VECIN, addr, MAX_BS * GROUP_SIZE);
        addr += (uint32_t)(MAX_BS * GROUP_SIZE * sizeof(T));

        // W buffer (2 for double buffering)
        int32_t w_packed_size = GROUP_SIZE * (TILE_N / PACK_RATIO);
        w_local[0] = LocalTensor<int32_t>(TPosition::VECIN, addr, w_packed_size);
        addr += (uint32_t)(w_packed_size * sizeof(int32_t));
        w_local[1] = LocalTensor<int32_t>(TPosition::VECIN, addr, w_packed_size);
        addr += (uint32_t)(w_packed_size * sizeof(int32_t));

        // S buffer (2 for double buffering)
        s_local[0] = LocalTensor<T>(TPosition::VECIN, addr, TILE_N);
        addr += (uint32_t)(TILE_N * sizeof(T));
        s_local[1] = LocalTensor<T>(TPosition::VECIN, addr, TILE_N);
        addr += (uint32_t)(TILE_N * sizeof(T));

        global_acc = LocalTensor<float>(TPosition::VECCALC, addr, MAX_BS * TILE_N);
        addr += (uint32_t)(MAX_BS * TILE_N * sizeof(float));

        // Acc buffers
        group_acc = LocalTensor<half>(TPosition::VECCALC, addr, MAX_BS * TILE_N);
        addr += (uint32_t)(MAX_BS * TILE_N * sizeof(half) + 256);

        w_half = LocalTensor<half>(TPosition::VECCALC, addr, GROUP_TILE * TILE_N);
        addr += (uint32_t)(GROUP_TILE * TILE_N * sizeof(half));
    }

    __aicore__ inline void ProcessTask(int32_t task_id) {
        int32_t n_tile_idx = task_id / split_k;
        int32_t k_split_idx = task_id % split_k;

        int32_t n_start = n_tile_idx * TILE_N;
        int32_t cur_n_len = (out_dim - n_start < TILE_N) ? (out_dim - n_start) : TILE_N;

        int32_t groups_per_split = (total_groups + split_k - 1) / split_k;
        int32_t k_group_start = k_split_idx * groups_per_split;
        int32_t k_group_end = (k_group_start + groups_per_split < total_groups) ? (k_group_start + groups_per_split) : total_groups;

        if (k_group_start >= total_groups) return;

        // Init Global Accumulator to 0
        for (int b = 0; b < batch_size; ++b) {
            Duplicate(global_acc[b * TILE_N], 0.0f, cur_n_len);
        }

        int ping = 0;
        PrefetchGroup(ping, k_group_start, n_start, cur_n_len);
        SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);

        for (int32_t g = k_group_start; g < k_group_end; ++g) {
            WaitFlag<HardEvent::MTE2_V>(EID_MTE2_V);
            if (g > k_group_start) {
                WaitFlag<HardEvent::V_MTE2>(EID_V_MTE2);
            }

            if (g + 1 < k_group_end) {
                int pong = ping ^ 1;
                PrefetchGroup(pong, g + 1, n_start, cur_n_len);
                SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);
            }

            ComputeGroup(ping, g, cur_n_len);

            if (g + 1 < k_group_end) {
                SetFlag<HardEvent::V_MTE2>(EID_V_MTE2);
            }
            ping ^= 1;
        }

        WriteOut(n_start, cur_n_len);
    }

    __aicore__ inline void PrefetchGroup(int buf_id, int group_idx, int n_start, int cur_n_len) {
        int32_t cur_n_packed = cur_n_len / PACK_RATIO;
        int32_t n_start_packed = n_start / PACK_RATIO;
        
        // 1. Fetch W for the group
        uint64_t w_offset = (uint64_t)group_idx * GROUP_SIZE * (uint64_t)out_dim_packed + n_start_packed;
        DataCopyExtParams p_w = {
            (uint16_t)GROUP_SIZE,
            (uint32_t)(cur_n_packed * sizeof(int32_t)),
            (uint32_t)((out_dim_packed - cur_n_packed) * sizeof(int32_t)),
            (uint32_t)((TILE_N / PACK_RATIO - cur_n_packed) * sizeof(int32_t) / 32),
            0
        };
        DataCopyPadExtParams<int32_t> pad_w{false, 0, 0, 0};
        DataCopyPad(w_local[buf_id], weightGm[w_offset], p_w, pad_w);

        // 2. Fetch Scale for the group
        uint64_t s_offset = (uint64_t)group_idx * out_dim + n_start;
        DataCopy(s_local[buf_id], scalesGm[s_offset], cur_n_len);

        // 3. Fetch X (batch_size rows) using DataCopyPad
        uint64_t x_offset = group_idx * GROUP_SIZE; // starting position for batch 0
        DataCopyExtParams p_x = {
            (uint16_t)batch_size,
            (uint32_t)(GROUP_SIZE * sizeof(T)),
            (uint32_t)((in_dim - GROUP_SIZE) * sizeof(T)), // distance between tail of batch b and head of batch b+1 in GM
            0, // contiguous in UB
            0
        };
        DataCopyPadExtParams<T> pad_x{false, 0, 0, 0};
        DataCopyPad(x_local[buf_id], xGm[x_offset], p_x, pad_x);
    }

    __aicore__ inline void ComputeGroup(int buf_id, int group_idx, int cur_n_len) {
        LocalTensor<half> x_half = x_local[buf_id].template ReinterpretCast<half>();
        
        // bf16 -> fp32 -> fp16 conversion
        if constexpr (IsBFloat16<T>::value) {
            LocalTensor<float> f_tmp = w_half.template ReinterpretCast<float>();
            Cast(f_tmp, x_local[buf_id], RoundMode::CAST_NONE, batch_size * GROUP_SIZE);
            Cast(x_half, f_tmp, RoundMode::CAST_ROUND, batch_size * GROUP_SIZE);
        }

        // Initialize group_acc per batch
        for (int b = 0; b < batch_size; ++b) {
            Duplicate(group_acc[b * TILE_N], (half)0.0f, cur_n_len);
        }

        // Dequantize and Mac
        for (int32_t k_inner = 0; k_inner < GROUP_SIZE; k_inner += GROUP_TILE) {
            LocalTensor<int4b_t> w_int4 = w_local[buf_id][k_inner * (TILE_N / PACK_RATIO)].template ReinterpretCast<int4b_t>();
            Cast(w_half, w_int4, RoundMode::CAST_NONE, GROUP_TILE * TILE_N);

            for (int b = 0; b < batch_size; ++b) {
                // Use uint64_t GetValue to load 4 halfs at once, reducing transactions
                LocalTensor<uint64_t> x_u64 = x_half.template ReinterpretCast<uint64_t>();
                half x_val_arr[GROUP_TILE];
                uint64_t* x_val_arr_u64 = (uint64_t*)x_val_arr;
                int x_base_idx = b * GROUP_SIZE + k_inner;
                x_val_arr_u64[0] = x_u64.GetValue(x_base_idx / 4);
                x_val_arr_u64[1] = x_u64.GetValue(x_base_idx / 4 + 1);

                for (int k = 0; k < GROUP_TILE; ++k) {
                    Axpy(group_acc[b * TILE_N], w_half[k * TILE_N], x_val_arr[k], cur_n_len);
                }
            }
        }

        // Accumulate to global float
        if constexpr (IsBFloat16<T>::value) {
            LocalTensor<float> f_acc = w_half.template ReinterpretCast<float>();
            LocalTensor<float> f_scale = f_acc[TILE_N];
            Cast(f_scale, s_local[buf_id], RoundMode::CAST_NONE, cur_n_len);

            for (int b = 0; b < batch_size; ++b) {
                Cast(f_acc, group_acc[b * TILE_N], RoundMode::CAST_NONE, cur_n_len);
                MulAddDst(global_acc[b * TILE_N], f_acc, f_scale, cur_n_len);
            }
        } else {
            for (int b = 0; b < batch_size; ++b) {
                MulAddDst(global_acc[b * TILE_N], group_acc[b * TILE_N], s_local[buf_id], cur_n_len);
            }
        }
    }

    __aicore__ inline void WriteOut(int n_start, int cur_n_len) {
        SetFlag<HardEvent::V_MTE3>(EID_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EID_V_MTE3);

        if (split_k > 1) {
            SetAtomicAdd<OutputT>();
        }

        // Stride write out using DataCopyPad
        DataCopyExtParams p_y = {
            (uint16_t)batch_size,
            (uint32_t)(cur_n_len * sizeof(OutputT)),
            (uint32_t)((TILE_N - cur_n_len) * sizeof(OutputT) / 32), // Packed consecutively in UB, so src stride between tail/head is 0
            (uint32_t)((out_dim - cur_n_len) * sizeof(OutputT)), // dst stride in GM in bytes
            0
        };

        // Output starts at n_start for batch 0
        uint64_t y_offset = n_start; 
        DataCopyPad(yGm[y_offset], global_acc, p_y);

        if (split_k > 1) {
            SetAtomicNone();
        }

        SetFlag<HardEvent::MTE3_V>(EID_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(EID_MTE3_V);
    }

private:
    GlobalTensor<T> xGm;
    GlobalTensor<int32_t> weightGm;
    GlobalTensor<T> scalesGm;
    GlobalTensor<OutputT> yGm;

    LocalTensor<T> x_local[2];
    LocalTensor<int32_t> w_local[2];
    LocalTensor<T> s_local[2];

    LocalTensor<half> group_acc;
    LocalTensor<float> global_acc;
    LocalTensor<half> w_half;

    int32_t batch_size = 0;
    int32_t in_dim = 0;
    int32_t out_dim = 0;
    int32_t out_dim_packed = 0;
    int32_t split_k = 1;
    int32_t total_groups = 0;
    int32_t num_n_tiles = 0;
    int32_t total_tasks = 0;
    static constexpr int32_t TILE_N = 2048;
};

// -----------------------------------------------------------------------------
// Batch GEMM W4A16 Small BS Impl (Vertical Fusion)
// -----------------------------------------------------------------------------
template<typename T>
__aicore__ inline void batch_gemm_w4a16_small_bs_impl(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scales,
    GM_ADDR y, GM_ADDR workspace,
    int32_t batch_size, int32_t in_dim, int32_t out_dim)
{
    // Phase 0: Zero out FP32 workspace (size = batch_size * out_dim)
    {
        uint64_t total_elems = (uint64_t)batch_size * (uint64_t)out_dim;
        KernelZeroOutLL<float> zeroOp;
        zeroOp.Init(workspace, total_elems);
        zeroOp.Process();
    }

    AscendC::SyncAll();

    // Phase 1: GEMM computation with FP32 output to workspace (atomic add)
    {
        KernelBatchGemmW4A16<T, float> op;
        op.Init(x, weight, scales, workspace, batch_size, in_dim, out_dim);
        op.Process();
    }

    AscendC::SyncAll();

    // Phase 2: Cast FP32 workspace to T output
    {
        KernelCastFP32ToT<T> op_cast;
        op_cast.Init(workspace, y, (int32_t)((uint64_t)batch_size * (uint64_t)out_dim));
        op_cast.Process();
    }
}

extern "C" __global__ __aicore__ void batch_gemm_w4a16_small_bs_fp16(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scales,
    GM_ADDR y, GM_ADDR workspace,
    int32_t batch_size, int32_t in_dim, int32_t out_dim)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::InitSocState();
    batch_gemm_w4a16_small_bs_impl<half>(x, weight, scales, y, workspace,
                                         batch_size, in_dim, out_dim);
}

extern "C" __global__ __aicore__ void batch_gemm_w4a16_small_bs_bf16(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scales,
    GM_ADDR y, GM_ADDR workspace,
    int32_t batch_size, int32_t in_dim, int32_t out_dim)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::InitSocState();
    batch_gemm_w4a16_small_bs_impl<bfloat16_t>(x, weight, scales, y, workspace,
                                               batch_size, in_dim, out_dim);
}

/*
llvm-objdump -t ./device_aiv.o | grep "fused_moe_small_bs_w4a16_fp16_18_mix_aiv"
000000000001fbec l       .text  0000000000000000 fused_moe_small_bs_w4a16_fp16_18_mix_aiv$local
000000000001fbec g     F .text  0000000000002290 fused_moe_small_bs_w4a16_fp16_18_mix_aiv
0000000000000048  w    O __CCE_KernelArgSize    0000000000000004 fused_moe_small_bs_w4a16_fp16_18_mix_aiv__
*/
extern "C" __global__ __aicore__ void fused_moe_small_bs_w4a16_fp16(
    GM_ADDR x,
    GM_ADDR w13_weight, GM_ADDR w13_scales,
    GM_ADDR w2_weight,  GM_ADDR w2_scales,
    GM_ADDR expert_ids, GM_ADDR topk_weights,
    GM_ADDR y,
    GM_ADDR workspace,
    int32_t batch_size, int32_t hidden_size, int32_t inter_size,
    int32_t num_experts, int32_t top_k)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::InitSocState();
    AscendC::ICachePreLoad(5);

    fused_moe_small_bs_impl<half>(x,
                                 w13_weight, w13_scales,
                                 w2_weight,  w2_scales,
                                 expert_ids, topk_weights,
                                 y, workspace,
                                 batch_size, hidden_size, inter_size,
                                 num_experts, top_k);
}

extern "C" __global__ __aicore__ void grouped_gemv_w4a16_moe_bf16(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scales,
    GM_ADDR expert_ids, GM_ADDR y,
    int32_t total_tokens, int32_t in_dim, int32_t out_dim,
    int32_t num_experts, int32_t top_k)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::InitSocState();

    KernelGroupedGemvW4A16Moe<bfloat16_t> op;
    op.Init(x, weight, scales, expert_ids, y, nullptr,
            total_tokens, in_dim, out_dim, num_experts, top_k, false, false);
    op.Process();
}

extern "C" __global__ __aicore__ void grouped_gemv_w4a16_moe_fp16(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scales,
    GM_ADDR expert_ids, GM_ADDR y,
    int32_t total_tokens, int32_t in_dim, int32_t out_dim,
    int32_t num_experts, int32_t top_k)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::InitSocState();

    KernelGroupedGemvW4A16Moe<half> op;
    op.Init(x, weight, scales, expert_ids, y, nullptr,
            total_tokens, in_dim, out_dim, num_experts, top_k, false, false);
    op.Process();
}