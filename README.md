# NR-CCP Demo

**Neural Risk-Aware Near-Optimal Coverage Path Planning** 的 2D 轻量演示原型。

> 说明：这是 **baseline demo**，不是完整 NR-RRT / Fields2Cover 复现。用于展示「长方形农田 + 压实 risk field + 覆盖路径规划 + 方法对比」。

## 项目要解决什么

在长方形农田上，机器需要覆盖全部可通行区域，同时尽量降低路径经过高 compaction risk 区域的代价。对比三种策略：

| 方法 | 说明 |
|------|------|
| `uniform_boustrophedon` | 标准之字形覆盖，忽略 risk |
| `risk_aware_reordering` | 同样条带顺序，高风险过渡段改走场边低 risk 通道 |
| `informed_strip_selection` | 优先访问高风险条带（informed prior） |

## 环境

```bash
cd ~/Desktop/NR_CCP_Demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行

```bash
# 默认配置
python main.py

# 或
python scripts/run_demo.py

# 条带宽度 batch 实验
python scripts/run_batch.py
```

输出：

- `outputs/figures/*.png` — 各方法路径图 + 对比柱状图
- `outputs/results/metrics.csv` — 数值指标

## 目录结构

```text
NR_CCP_Demo/
  configs/default.yaml    # 全部参数
  src/
    config_loader.py      # 配置解析
    geometry.py           # 农田网格
    fields.py             # risk field 生成
    planner.py            # 三种覆盖规划器
    metrics.py            # 路径指标
    visualization.py      # 出图
  main.py                 # 主入口
  scripts/                # 便捷脚本
  outputs/                # 运行产物（git 可忽略）
```

## 图该怎么讲（演示 3 分钟）

1. **背景**：农田覆盖路径规划（CPP）不是点对点最短路，要覆盖整块区域。
2. **Risk field**：深色区域 = 土壤压实/重复碾压风险高（由 Gaussian hotspot 模拟）。
3. **Uniform**：从左到右扫，简单但可能在高低风险区之间来回折返。
4. **Risk-aware**：条带顺序变了，risk×length 通常更低。
5. **Informed**：先处理高风险条带，体现 informed prior 思想。
6. **局限**：2D 栅格、无 NR-RRT 学习模块、无 Fields2Cover；下一步可接真实 benchmark。

## 配置说明

编辑 `configs/default.yaml`：

- `field` — 农田尺寸、网格分辨率、禁行区
- `risk_field.gaussians` — 风险热点位置与强度
- `planner.strip_width_m` — 覆盖条带宽度（≈ 作业幅宽）
- `methods` — 要对比的方法列表

## 指标

- `path_length_m` — 总路径长度
- `risk_length_cost` — 路径段长度 × 段平均 risk 之和
- `mean_risk` / `max_risk` — 路径点 risk 统计
- `coverage_rate` — 条带半径内被访问到的自由栅格比例
