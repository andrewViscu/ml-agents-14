# ML-Agents Prediction Models

Prediction models for SoccerTwos and Worm environments in Unity ML-Agents.

## Research Questions

**RQ1:** Can resource usage (memory, CPU, GPU) be predicted from hyperparameters?

**RQ2:** Can training reward be predicted from hyperparameters?

## Quick Start

open the PredictionsModels directory and then by running these scripts you can get the results used in the report and readme. 

### RQ1: Resource Prediction

```bash
python rq1_resource_prediction.py --training_data ../training-data
```

### RQ2: Reward Prediction

```bash
python rq2_hyperparameter_prediction.py --training_data ../training-data --shared_data ../shared-data
```

## Results

### RQ1: Resource Prediction

**SoccerTwos** (4 runs, 1750 data points)

| Resource | R² | MAE | <10% | <25% |
|----------|-----|-----|------|------|
| Memory (MB) | -0.92 | 253.31 | 24.2% | 64.6% |
| CPU (%) | 0.82 | 0.12 | 21.1% | 52.8% |
| GPU (MB) | -1.20 | 552.16 | 0.0% | 0.0% |

**Worm** (22 runs, 1697 data points)

| Resource | R² | MAE | <10% | <25% |
|----------|-----|-----|------|------|
| Memory (MB) | 0.95 | 24.31 | 95.4% | 99.9% |
| CPU (%) | 0.21 | 0.39 | 17.7% | 44.1% |
| GPU (MB) | 0.38 | 106.86 | 61.6% | 86.2% |

### RQ2: Reward Prediction

| Game | Runs | R² | MAE | <25% | <50% |
|------|------|-----|-----|------|------|
| SoccerTwos | 142 | 0.08 | 0.0203 | 53.5% | 78.2% |
| Worm | 40 | 0.69 | 107.56 | 27.5% | 40.0% |

## Key Findings

- Worm memory prediction: R²=0.95 (95.4% within 10% error)
- SoccerTwos CPU prediction: R²=0.82
- Worm reward prediction: R²=0.69
- SoccerTwos reward is not predictable (R²=0.08) due to narrow reward range

## Files

```
PredictionModels/
├── rq1_resource_prediction.py
├── rq2_hyperparameter_prediction.py
├── utils/
│   ├── cross_validation.py
│   ├── step_level_resource_loader.py
│   └── hyperparameter_data_loader.py
└── extra/
    ├── cutoff_analysis.py
    └── all_games_reward_prediction.py
```

## Model

Ensemble: Gradient Boosting (70%) + Ridge Regression (30%)

Cross-validation:
- RQ1: GroupKFold (5 folds, grouped by run)
- RQ2: KFold (5 folds)

## Data

- `training-data/`: Step-level CSV files with resource metrics
- `shared-data/multi_run/only_soccertwo.csv`: SoccerTwos runs
- `shared-data/single_run/`: Worm static config and summary
