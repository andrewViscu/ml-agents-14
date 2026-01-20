# Project 2-1 Group 14

This document outlines the instructions for running our data collection scripts, documentation of the metrics that are calculated and stored, and instructions on how to run the prediction model.

In this repository, the data collection scripts are located in the `data-extraction-scripts` package. The prediction model files are in the `PredictionModels` package.

---

# Running Data Collection Scripts

## Requirements

The following libraries (excluding those required by the ML-Agents library) and software versions are required to run the scripts.

1- Install unity editor, version: 2022.3.14f1
2- Install python and anaconda
3- create a python virtual environment (venv) using

```bash
“conda create -n mlagents python=3.10.8 && conda activate mlagents
```

4- Build the project wheel using:

```bash
“conda install "grpcio=1.48.2" -c conda-forge” 
```

5-  ⁠⁠install mlagents (inside this venv) by using

```bash
“python -m pip install mlagents==1.1.0”
```

6- Install all required libraries using

```bash
pip install -r requirements.txt
```

## Instructions

In order to train the models to collect data without having to open Unity every single run, an executable for your desired game must be created. The following tutorial can be used to create executables:

- <https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/Learning-Environment-Executable.html>

>[!IMPORTANT]
>Executables should be stored in the `training-envs` folder. For consistency, create a new appropriately named folder inside `training-envs` for each game.

After an executable has been created, activate your python 3.10 virtual environment and run the following command from the root of the project to run the data collection scripts.

```bash
python ./data-extraction-scripts/data_extraction_script.py <config_file_path> <training-env_path> <run_id>
```

where:

- `<config file path>` is the path to the `.yaml` file with hyperparameters of the game you want to collect data for.
- `<training-env_path>` is the path to the executable for the chosen game.
- `<run_id>` is the name ID for the training run.

>[!NOTE]
>Config files are stored in the `unity/config` package. We decided to store the ML-Agents library files there for a cleaner developer experience.

Our custom config files can be found in the `unity/config/custom` package.

>[!WARNING]
>Metrics from tensorboard such as policy loss, entropy, and value loss are recorded at the end of a training run i.e. after the config file has reached it's `max_steps` limit. For example, if a config file has `max_steps` set at 2 million, the tensorboard metrics will be recorded only if the model is trained for 2 million steps. That way, training can automatically finish on it's own.

## Data storage

The generated CSV table from the training run will be stored in the `training_data` package (which you can find from the root) and will be named `training_data_<run_id>.csv`, provided the `run_id`.

### Example Run

```bash
python ./data-extraction-scripts/data_extraction_script.py unity/config/custom/Worm_Cai.yaml training-envs/worm-windows/UnityEnvironment WORM_RUN1
```

---

# Data Metrics Documentation

The list below contains the metrics that the scripts calculate and store. The names of the columns in the CSV table are the same as these:

Model Performance (ML-Agents provided):

- **step** - The number of steps performed.
- **mean_reward** - The mean reward for the current run.
- **mean_group_reward** - The mean reward of the group for the current run, applicable only to multi-agent environments.
- **std_reward** - The standard deviation of the reward.
- **time_elapsed** - The time elapsed of the current run.

Resource Usage:

- **memory_usage_avg_mb** - Average memory usage in MB.
- **memory_usage_peak_mb** - Peak memory usage in MB.
- **cpu_usage_avg_percent** - Average CPU usage in percent.
- **cpu_usage_peak_percent** - Peak CPU usage in percent.
- **gpu_usage_avg_mb** - Average GPU usage in MB (can not be calculated if your system has an integrated graphics card).
- **gpu_usage_peak_mb** - Peak GPU usage in MB.

Tensorboard provided performance metrics:

- **policy_entropy** - Policy entropy.
- **losses_policy_loss** - Policy loss.
- **losses_value_loss** - Value loss.

Recorded hyperparameter values:

- **trainer_type** - The algorithm used by the agent.
- **batch_size**
- **buffer_size**
- **learning_rate**
- **beta**
- **epsilon**
- **lambd**
- **num_epoch**
- **learning_rate_schedule**

---

# Machine Learning Model

Prediction models for SoccerTwos and Worm environments in Unity ML-Agents.

## Research Questions

**RQ1:** Can resource usage (memory, CPU, GPU) be predicted from hyperparameters?

**RQ2:** Can training reward be predicted from hyperparameters?

## Quick Start

open the PredictionsModels directory and then by running these scripts you can get the results used in the report and readme.

### RQ1: Resource Prediction

```bash
python PredictionModels/rq1_resource_prediction.py --training_data training-data
```

### RQ2: Reward Prediction

```bash
python PredictionModels/rq2_hyperparameter_prediction.py --training_data training-data --shared_data shared-data
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

```bash
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

---

## Extra Analysis Scripts

Additional scripts exploring alternative approaches and extended analysis beyond the main RQ1/RQ2.

### Scripts

### cutoff_analysis.py

Tests the original RQ2 approach: predicting final reward from early training data at different step cutoffs.

```bash
python PredictionModels/extra/cutoff_analysis.py --training_data training-data --shared_data shared-data
```

**Results (423 runs with time-series data):**

| Cutoff | Runs | R² |
|--------|------|-----|
| 100,000 | 423 | 0.10 |
| 200,000 | 423 | 0.21 |
| 300,000 | 338 | 0.25 |
| 400,000 | 337 | 0.37 |
| 500,000 | 227 | 0.45 |
| 600,000 | 225 | 0.51 |
| 800,000 | 225 | 0.56 |
| 1,000,000 | 131 | 0.73 |

More early training data improves predictions, but requires waiting longer before prediction is useful.

### all_games_reward_prediction.py

Extends RQ2 to all available games to test if single-agent vs multi-agent environments differ in predictability.

```bash
python PredictionModels/extra/all_games_reward_prediction.py --shared_data shared-data
```

**Results:**

| Game | Type | Runs | R² | Reward Range |
|------|------|------|-----|--------------|
| 3DBall | Single | 4390 | 0.77 | 99.2 |
| GridFoodCollector | Multi | 198 | 0.56 | 72.0 |
| PushBlock | Multi | 192 | 0.53 | 6.0 |
| Crawler | Single | 356 | 0.42 | 2195.4 |
| Basic | Single | 46 | 0.14 | 0.8 |
| SoccerTwos | Multi | 142 | 0.08 | 0.1 |
| Hallway | Multi | 49 | -0.33 | 1.4 |

**Excluded:** BigWallJump (no HP variation), GridWorld/Pyramids/Sorter/Walker/Worm (< 30 runs)

**Findings:**

Predictability depends on:

1. **Sample size** - need 30+ runs for reliable results
2. **Reward range** - narrow ranges (SoccerTwos: 0.1, Hallway: 1.4) yield poor R²
3. **HP variation** - identical hyperparameters make prediction impossible

Single vs multi-agent distinction is not the determining factor.
