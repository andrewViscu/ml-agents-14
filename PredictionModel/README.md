# ML-Agents Training Predictor

## Group 14 - BCS2720 Project 2.1

Predicts ML agent performance and compares training algorithms (PPO, SAC, POCA) using data collected from Unity ML-Agents environments.

## Usage

### Basic usage (run all analyses):

```bash
python main.py --data_dir ./data
```

## What Each Mode Does

### reward

Predicts if a training step will have good or bad reward based on resource metrics.
Uses random forest with 5-fold cross validation.

### algorithm

Predicts which algorithm (PPO/SAC/POCA) is being used.
Framework for comparing algorithm performance.
**Note: needs data from multiple algorithms to be useful**

### environment

Analyzes performance across different environments (3DBall, SoccerTwos, etc).
Recommends best algorithm for each environment.

### convergence

| Target | Runs | R-squared | MAPE | Notes |
|--------|------|-----------|------|-------|
| Training Time | 3,913 | 0.54 | 41% | From hyperparameters only |
| CPU Usage | 1,462 | 0.79 | 41% | With hardware specs |
| RAM Usage | 1,462 | 0.62 | 23% | With hardware specs |
| CPU Usage | 3,913 | 0.09 | 13% | Without hardware info |
| RAM Usage | 3,913 | 0.04 | 16% | Without hardware info |

### Performance Prediction (RQ2) - Final Reward

**Per-Game Results (training-data, 500k cutoff):**

| Game | Runs | R-squared | Reward Range |
|------|------|-----------|--------------|
| Worm | 22 | 0.48 | 17 - 1207 |
| SoccerTwos | 4 | -1.39 | -0.09 - 0.10 |
| Combined | 26 | 0.56 | - |

The combined R2 (0.56) is misleading because Worm's larger reward variance dominates the metric. SoccerTwos has a negative R2 due to insufficient samples (4 runs) and a narrow reward range.

**Effect of Early Data Cutoff (combined dataset):**

| Cutoff | Runs | R-squared |
|--------|------|-----------|
| 100,000 steps | 1,192 | 0.26 |
| 200,000 steps | 1,176 | 0.31 |
| 300,000 steps | 919 | 0.44 |
| 400,000 steps | 918 | 0.52 |
| 500,000 steps | 640 | 0.66 |

Key findings:

- Different games have incompatible reward scales, making combined prediction unreliable
- Per-game models are more meaningful but require sufficient samples per game
- More early training data improves predictions (R2: 0.26 at 100k to 0.66 at 500k)

## Structure

``` bash
PredictionModel/
    performance_predictor.py  # CLI for performance prediction
    resource_predictor.py     # CLI for resource prediction
    models/
        base_predictor.py     # Shared ML functionality
        performance_model.py  # Performance prediction model
        resource_model.py     # Resource prediction model
    utils/
        data_loader.py        # Data loading utilities
```

## Data Format

Required CSV columns:

- step (or steps, training_step)
- A reward column (mean_reward, mean_group_reward)

Optional:

- run_id, environment, algorithm
- Resource metrics (cpu_usage_percent, ram_usage_mb)
- Hyperparameters (learning_rate, batch_size)
- Hardware info (cpu_model, gpu_model, total_ram)
