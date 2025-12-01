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
Checks if training runs are converging based on reward trends.

## Data Format

CSV files should have these columns:
- `step` - training step number
- `mean_reward` - individual agent reward
- `mean_group_reward` - team/group reward
- `run_id` - unique identifier for the run
- `environment` - environment name (e.g., 'multi-agent', '3dball')
- `algorithm` - algorithm used ('ppo', 'sac', 'poca')
- `memory_usage_avg_mb` - average memory usage
- `memory_usage_peak_mb` - peak memory usage
- `cpu_usage_avg_mb` - average CPU usage
- `cpu_usage_peak_mb` - peak CPU usage
- `gpu_usage_avg_mb` - average GPU usage (optional)
- `gpu_usage_peak_mb` - peak GPU usage (optional)

## Adding New Data

1. Run training with your data collection scripts
2. Save CSV files to the `data/` directory
3. Make sure column names match the format above
4. Run `python main.py` to analyze

## TODO

- Add more training runs from different algorithms (PPO, SAC)
- Add data from different environments (3DBall, Hallway, StrikerVsGoalie)
- Implement early stopping prediction
- Add visualization/plots
- Save trained models for later use

## Current Results

With soccer_twos POCA data only:
- Reward prediction CV accuracy: ~69.7%
- Algorithm prediction: needs more algorithms

## Dependencies

- pandas
- numpy
- scikit-learn
