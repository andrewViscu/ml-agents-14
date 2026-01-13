# Project 2-1 Group 14

This document outlines the instructions for running our data collection scripts and includes documentation of the metrics that are calculated and stored.

---

# Running Data Collection Scripts

## Requirements

The following libraries (excluding those required by the ML-Agents library) and software versions are required to run the scripts.

```bash
Run pip install -r requirements.txt
```

## Instructions

In order to train the models to collect data without having to open Unity every single run, an executable for your desired game must be created. The following tutorial can be used to create executables:

- <https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/Learning-Environment-Executable.html>

>[!IMPORTANT]
>Executables must be stored in the `training-envs` folder. For consistency, create a new appropriately named folder inside `training-envs` for each game.

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

- **step** - The number of steps performed.
- **mean_reward** - The mean reward for the current run.
- **mean_group_reward** - The mean reward of the group for the current run, applicable only to multi-agent environments.
- **std_reward** - The standard deviation of the reward.
- **time_elapsed** - The time elapsed of the current run.
- **memory_usage_avg_mb** - Average memory usage in MB.
- **memory_usage_peak_mb** - Peak memory usage in MB.
- **cpu_usage_avg_percent** - Average CPU usage in percent.
- **cpu_usage_peak_percent** - Peak CPU usage in percent.
- **gpu_usage_avg_mb** - Average GPU usage in MB (can not be calculated if your system has an integrated graphics card).
- **gpu_usage_peak_mb** - Peak GPU usage in MB.
- **trainer_type** - The algorithm used by the agent.
- **policy_entropy** - Policy entropy.
- **losses_policy_loss** - Policy loss.
- **losses_value_loss** - Value loss.

Recorded hyperparameter values:

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