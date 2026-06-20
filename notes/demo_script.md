# NR-CCP 演示讲稿（约 3 分钟）

## 30 秒版

我在做 NR-CCP：把王组 NR-RRT 的 risk-bounded 思想迁移到覆盖路径规划。这个 demo 在 2D 农田上生成 compaction risk field，对比普通之字形覆盖、risk-aware 条带重排、以及 informed 条带优先三种 baseline，用路径长度和 risk×length 量化效果。

## 展开版

1. **问题**：农业机器人要覆盖整块田，不是从 A 到 B。重复碾压会压实土壤，所以路径还要考虑 risk。
2. **输入**：100m×60m 长方形农田 + 多个高斯 risk 热点。
3. **方法**：
   - Uniform：标准 boustrophedon，从左到右扫。
   - Risk-aware：条带顺序按 transition cost + strip risk 贪心优化。
   - Informed：高风险条带优先访问。
4. **结果**：打开 `outputs/figures/comparison.png`，指出 risk×length 差异。
5. **诚实说明**：这是 V1 baseline，未接 NR-RRT 神经网络和 Fields2Cover；下一步接真实农田 benchmark。

## 教授可能追问

- **和 NR-RRT 什么关系？** NR-RRT 是点对点 risk-bounded sampling；NR-CCP 把 risk 写进覆盖目标。
- **为什么不用 Fields2Cover？** demo 阶段先用自写 2D 条带覆盖验证思路，后续可替换为 Fields2Cover headland/route。
- **informed 和 risk-aware 区别？** risk-aware 最小化 traversal cost；informed 用 risk 先验决定条带优先级。
