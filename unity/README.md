# Project 2-1 Group 14

This document outlines the instructions for running our data collection scripts and documentation on the metrics that are calculated and stored.
# Running Data Collection Scripts
### Requirements
The following libraries and software versions are required to run the scripts.

Software:
- Unity version - 2023.2.12f1
- Python - 3.10.x

Libraries:
- psutil
- pynvml

### Instructions
Open your desired game in the Unity Editor.

Activate your python 3.10 virtual environment and run the following command from the root of the project to run the data collection scripts.
```
python data-extraction-scripts/data_extraction_script.py <config file path> <run_id_name>
```
where:
- ```<config file path>``` is the path to the ```.yaml``` file of the game you want to collect data for.
- ```<run_id_name>``` is the ID for the training run.

**Example:**
```
python data-extraction-scripts/data_extraction_script.py config/ppo/3DBall.yaml example_run
```

The program will prompt you to run the specified game in the Unity Editor:
```
[INFO] Listening on port 5004. Start training by pressing the Play button in the Unity Editor.
```

Press Play in the Unity Editor for your specified game and the collection process will begin.
### Data storage
The generated CSV table from the training run will be stored in the ```training_data``` package (which you can find from the root) and will be named ```training_data.csv``` by default.

---
# Data Metrics Documentation
The list below contains the metrics that the scripts calculate and store. The names of the columns in the CSV table are the same as these:
- **step** - The number of steps performed
- **mean_reward** - The mean reward of the current run
- **std_reward** - Standard deviation of the cumulative reward
- **timestamp** - The time stamp of the training run
- **run_id** - The run ID of the training session (used for distinction during custom machine learning model training)
- **environment** - The environment in which the agent trains in (single or multi-agent)
- **algorithm** - The algorithm the agent is using
- **memory_usage_avg_mb** - Average memory usage in MB
- **memory_usage_peak_mb** - Peak memory usage in MB
- **cpu_usage_avg_mb** - Average CPU usage in MB
- **cpu_usage_peak_mb** - Peak CPU usage in MB
- **gpu_usage_avg_mb** - Average GPU usage in MB
- **gpu_usage_peak_mb** - Peak GPU usage in MB
