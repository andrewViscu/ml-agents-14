# Project 2-1 Group 14

This document outlines the instructions for running our data collection scripts and includes documentation of the metrics that are calculated and stored.

---

# Running Data Collection Scripts
### Requirements
The following libraries (excluding those required by the ML-Agents library) and software versions are required to run the scripts.

Software:
- Unity version - 2023.2.12f1
- Python - 3.10.x

Libraries:
- psutil
- pynvml

### Instructions
In order to train the models to collect data without having to open Unity every single run, an executable for your desired game must be created. The following tutorial can be used to create executables: 
- https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/Learning-Environment-Executable.html

>[!IMPORTANT]
>Executables must be stored in the `training-envs` folder. For consistency, create a new appropriately named folder inside `training-envs` for each game.

After an executable has been created, activate your python 3.10 virtual environment and run the following command from the root of the project to run the data collection scripts.

```
python ./data-extraction-scripts/data_extraction_script.py <config_file_path> <training-env_path> <run_id>
```

where:
- `<config file path>` is the path to the `.yaml` file with hyperparameters of the game you want to collect data for.
- `<training-env_path>` is the path to the executable for the chosen game.
- `<run_id>` is the name ID for the training run.

>[!NOTE]
>Config files are stored in the `unity/config` package. We decided to store the ML-Agents library files there for a cleaner developer experience.

Our custom config files can be found in the folders inside the `config` package and are named the following:
**PPO:**
- Worm_Cai
- Worm_Schulman_MuJoCo
- Worm_Schulman_Roboschool
- Worm_Zhang
**SAC**
- Worm_Haarnoja

**Example run:**
```
python ./data-extraction-scripts/data_extraction_script.py unity/config/ppo/Worm.yaml training-envs/worm-windows/UnityEnvironment WORM_RUN1
```

### Data storage
The generated CSV table from the training run will be stored in the `training_data` package (which you can find from the root) and will be named `training_data_<run_id>.csv`, provided the `run_id`.

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
