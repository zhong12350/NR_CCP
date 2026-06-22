# NR-CCP

Neural Risk-Aware Coverage Path Planning (NR-CCP) is a lightweight Python prototype for risk-aware agricultural coverage path planning.

The project evaluates coverage routes on 2D field boundaries from WKT files. It builds a compaction risk field, generates coverage path candidates, and compares several planning/selection strategies under path length, risk, coverage, and risk-bound violation metrics.

> This repository is a research demo/prototype, not a full production implementation of NR-RRT or Fields2Cover.

## Features

- Load real field boundaries from WKT files.
- Generate headland-aware 2D field grids.
- Build synthetic soil compaction risk fields from headland distance, pass-count effects, repeated traversal penalties, and turning penalties.
- Compare multiple coverage planning baselines:
  - `naive`
  - `weighted`
  - `rb_ccp`
  - `nr_ccp`
  - `fields2cover`
- Run single-field demos, batch experiments, ablation studies, delta sweeps, and informed sampler training.
- Export figures and CSV metrics for later analysis.

## Repository Structure

```text
NR_CCP/
  configs/                 # Experiment configuration files
  data/                    # Small demo field data
  imgs/                    # Field preview images
  models/                  # Placeholder for trained sampler weights
  scripts/                 # Convenience experiment scripts
  src/                     # Core planning, risk, metrics, and visualization code
  wkt/                     # Field boundary WKT files
  main.py                  # Main entry point
  requirements.txt         # Python dependencies
```

Generated experiment results are written to `outputs/`, which is intentionally ignored by Git.

## Installation

```bash
git clone https://github.com/zhong12350/NR_CCP.git
cd NR_CCP

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies:

- NumPy
- Matplotlib
- PyYAML
- Shapely

## Quick Start

Run the default single-field experiment:

```bash
python main.py
```

Run a specific WKT field:

```bash
python main.py wkt/ee_field_10.wkt
```

Run the convenience demo script:

```bash
python scripts/run_demo.py
```

## Experiment Modes

```bash
# Run batch experiments over WKT fields
python main.py batch

# Generate advisor demo figures
python main.py advisor

# Analyze batch results and generate summary plots
python main.py analyze

# Run delta sensitivity experiments
python main.py delta_sweep

# Run ablation experiments
python main.py ablation

# Train the informed sampler
python main.py train_sampler
```

The default full experiment configuration is:

```text
configs/nr_ccp_full.yaml
```

You can also pass a custom YAML configuration:

```bash
python main.py configs/default.yaml
```

## Methods

The current implementation compares candidate selection rules on shared or informed candidate pools:

| Method | Description |
| --- | --- |
| `naive` | Selects the shortest feasible coverage path. |
| `weighted` | Minimizes path length plus weighted compaction cost. |
| `rb_ccp` | Risk-bounded coverage path planning on the full candidate pool. |
| `nr_ccp` | Applies the risk-bounded rule on an informed-search candidate pool. |
| `fields2cover` | Fields2Cover-like shortest-route baseline on the shared candidate pool. |

## Outputs

By default, results are saved under:

```text
outputs/
  figures/                 # Path visualizations and comparison plots
  results/                 # Per-field and batch CSV metrics
  advisor_demo/            # Advisor demo figures
```

Important metrics include:

- `path_length_m`: total path length.
- `compaction_cost`: accumulated risk-weighted traversal cost.
- `mean_risk`: mean risk along the selected path.
- `max_risk`: maximum risk along the selected path.
- `coverage_rate`: covered free-space ratio.
- `fallback`: whether the risk bound could not be satisfied.
- `violation`: amount by which the selected path exceeds the risk bound.

## Configuration

Most experiment settings are controlled by YAML files in `configs/`.

Common parameters:

- `field.wkt_path`: default field boundary file.
- `headland.width_m`: headland width.
- `risk_field`: risk model and penalty settings.
- `planner.swath_width_m`: coverage swath width.
- `planner.angle_step_deg`: candidate angle enumeration resolution.
- `selection.delta`: risk bound.
- `methods`: list of methods to compare.
- `batch.field_glob`: WKT files used in batch mode.

## Notes

- This is a compact research prototype for exploring risk-aware coverage path planning ideas.
- The included WKT and image files are used for reproducible demos and batch experiments.
- Trained model weights such as `models/*.npz` are ignored by Git and should be regenerated or shared separately if needed.

