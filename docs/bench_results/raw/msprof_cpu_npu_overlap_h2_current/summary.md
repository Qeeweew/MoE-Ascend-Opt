# msprof：当前双 stream CPU/NPU overlap（2 hit / 6 miss）

配置为 Qwen3 TP2、每节点 20 线程、BF16、TopK=8。初始化后 warmup 1 次，再只执行 1 次
目标调用。目标路径严格复刻当前 cache method：CPU callback 位于 side stream，cached NPU
fused-MoE 位于 main stream，最后 event join。

msprof 会放大 host/runtime 开销，因此脚本报告的 0.807 ms wall time不作为正常运行延迟；
重叠关系直接读取 device task timeline。

## 目标调用时间线

| 任务 | Stream | 开始时间（相对） | 时长 |
|---|---:|---:|---:|
| side D2H（4 个小 copy） | 44 | 0 us | 约 22.7 us（含间隔） |
| CPU callback 等待/执行窗口 | 44 | 35.6 us | 269.924 us |
| cached fused-MoE BF16 | 46 | 161.1 us | 41.002 us |
| main stream 等待 CPU | 46 | 210.5 us | 98.700 us |
| side H2D CPU output | 44 | 305.7 us | 2.880 us |
| 最终 Add | 46 | 361.1 us | 2.060 us |

绝对时间以 side stream 第一个 D2H task 的开始作为 0。NPU fused-MoE 的完整区间位于 CPU
callback 的 269.924 us 窗口内部：NPU 在 CPU 开始后约 125.4 us 启动，运行 41.002 us，
结束时 CPU 仍剩约 103.5 us。因此在 2 hit / 6 miss 下，**NPU device 计算被 CPU 完全隐藏**，
当前关键路径仍是 CPU callback；NPU 完成后 main stream 继续等待 CPU 约 98.7 us。

## 对单 stream 双 callback 方案的含义

建议顺序：

```text
D2H input
callback_start: 将 execute_fn 投递到独立 coordinator，立即返回
cached NPU fused-MoE
callback_join: 等待 coordinator 完成，立即返回
H2D CPU output
Add
```

H2D 应预先排在 callback_join 后面，不应由 callback 内部调用 ACL runtime。CPU coordinator
必须独立于现有 NUMA GEMV worker pool；`execute_fn` 内部还会向 NUMA launcher/thread pool 提交
任务并等待，如果 coordinator 占用同一个 worker pool，存在自锁风险。

原始文件：`timeline.json`、`task_time.csv`、`op_summary.csv`、`api_statistic.csv`、
`profile.db`。
