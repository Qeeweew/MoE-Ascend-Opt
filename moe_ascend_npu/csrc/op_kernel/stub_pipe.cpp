#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
class StubPipeOp {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t len) {
        tileLen = len / GetBlockNum() / BUFFER_NUM;
        xGm.SetGlobalBuffer((__gm__ half*)x + tileLen * GetBlockIdx(), tileLen);
        yGm.SetGlobalBuffer((__gm__ half*)y + tileLen * GetBlockIdx(), tileLen);
        pipe.InitBuffer(inQ, BUFFER_NUM, tileLen * sizeof(half));
        pipe.InitBuffer(outQ, BUFFER_NUM, tileLen * sizeof(half));
    }
    __aicore__ inline void Process() {
        for (int i = 0; i < BUFFER_NUM * 2; i++) {
            LocalTensor<half> xLocal = inQ.AllocTensor<half>();
            DataCopy(xLocal, xGm[i * tileLen], tileLen);
            inQ.EnQue(xLocal);
            LocalTensor<half> xDeq = inQ.DeQue<half>();
            LocalTensor<half> zLocal = outQ.AllocTensor<half>();
            Add(zLocal, xDeq, xDeq, tileLen);
            outQ.EnQue(zLocal);
            inQ.FreeTensor(xDeq);
            LocalTensor<half> zDeq = outQ.DeQue<half>();
            DataCopy(yGm[i * tileLen], zDeq, tileLen);
            outQ.FreeTensor(zDeq);
        }
    }
private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQ;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQ;
    GlobalTensor<half> xGm, yGm;
    uint32_t tileLen;
};
extern "C" __global__ __aicore__ void stub_pipe(GM_ADDR x, GM_ADDR y, uint32_t len) {
    StubPipeOp op;
    op.Init(x, y, len);
    op.Process();
}
