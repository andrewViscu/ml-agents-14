#!/usr/bin/env python3
import sys
import os
import re
import yaml
from csv_table_creator import create_csv_table
from training_runner import run_training
from tensorboard_metrics import TensorBoardMetrics


def parse_config_file(config_file_path):
    """
    Parse a YAML config file and extract hyperparameters.

    Args:
        config_file_path (str): Path to the YAML config file.

    Returns:
        dict: Dictionary containing extracted hyperparameters.
    """
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Initialize with default empty values
        hyperparams = {
            "trainer_type": "",
            "batch_size": "",
            "buffer_size": "",
            "learning_rate": "",
            "beta": "",
            "epsilon": "",
            "lambd": "",
            "num_epoch": "",
            "learning_rate_schedule": "",
            "buffer_init_steps": "",
            "tau": "",
            "steps_per_update": "",
            "save_replay_buffer": "",
            "init_entcoef": "",
            "reward_signal_steps_per_update": "",
        }

        # Extract behavior config
        if config and "behaviors" in config:
            behavior_name = list(config["behaviors"].keys())[0]
            behavior_config = config["behaviors"][behavior_name]

            # Extract trainer type
            hyperparams["trainer_type"] = behavior_config.get(
                "trainer_type", ""
                )

            # Extract hyperparameters
            if "hyperparameters" in behavior_config:
                hp = behavior_config["hyperparameters"]
                hyperparams["batch_size"] = hp.get("batch_size", "")
                hyperparams["buffer_size"] = hp.get("buffer_size", "")
                hyperparams["learning_rate"] = hp.get("learning_rate", "")
                hyperparams["beta"] = hp.get("beta", "")
                hyperparams["epsilon"] = hp.get("epsilon", "")
                hyperparams["lambd"] = hp.get("lambd", "")
                hyperparams["num_epoch"] = hp.get("num_epoch", "")
                hyperparams["learning_rate_schedule"] = hp.get(
                    "learning_rate_schedule", ""
                    )
                hyperparams["buffer_init_steps"] = hp.get(
                    "buffer_init_steps", ""
                    )
                hyperparams["tau"] = hp.get(
                    "tau", ""
                    )
                hyperparams["steps_per_update"] = hp.get(
                    "steps_per_update", ""
                    )
                hyperparams["save_replay_buffer"] = hp.get(
                    "save_replay_buffer", ""
                    )
                hyperparams["init_entcoef"] = hp.get(
                    "init_entcoef", ""
                    )
                hyperparams["reward_signal_steps_per_update"] = hp.get(
                    "reward_signal_steps_per_update", ""
                    )

            # Extract network settings
            if "network_settings" in behavior_config:
                ns = behavior_config["network_settings"]
                hyperparams["hidden_units"] = ns.get("hidden_units", "")
                hyperparams["num_layers"] = ns.get("num_layers", "")

            # Extract reward signals (gamma)
            if "reward_signals" in behavior_config and "extrinsic" in (
                behavior_config["reward_signals"]
            ):
                hyperparams["gamma"] = (
                    behavior_config["reward_signals"]["extrinsic"].get(
                        "gamma", ""
                    )
                )

            # Extract time_horizon
            hyperparams["time_horizon"] = behavior_config.get(
                "time_horizon", ""
                )

        return hyperparams

    except Exception as e:
        print(
            f"Warning: Could not parse config file '{config_file_path}': {e}"
            )
        # Return empty dict
        return {
            "trainer_type": "",
            "batch_size": "",
            "buffer_size": "",
            "learning_rate": "",
            "beta": "",
            "epsilon": "",
            "lambd": "",
            "num_epoch": "",
            "learning_rate_schedule": "",
            "buffer_init_steps": "",
            "tau": "",
            "steps_per_update": "",
            "save_replay_buffer": "",
            "init_entcoef": "",
            "reward_signal_steps_per_update": "",
        }


def parse_mlagents_output(line):
    """
    Parse a line from mlagents-learn output to extract training metrics.

    Args:
        line (str): A line of output from mlagents-learn command.

    Returns:
        dict: Dictionary containing extracted metrics:
            - step: Training step (int or None)
            - mean_reward: Mean reward (float or None)
            - std_reward: Standard deviation of reward (float or None)
            - mean_group_reward: Mean Group reward (float or None)
    """
    metrics = {
        "step": None,
        "mean_reward": None,
        "std_reward": None,
        "mean_group_reward": None,
        "time_elapsed": None,
    }

    # Extract step
    step_match = re.search(r"Step:\s*(\d+)", line, re.IGNORECASE)
    if step_match:
        metrics["step"] = int(step_match.group(1))

    # Extract mean reward
    mean_reward_match = re.search(
        r"Mean Reward:\s*([+-]?\d*\.?\d+)", line, re.IGNORECASE
    )
    if mean_reward_match:
        metrics["mean_reward"] = float(mean_reward_match.group(1))

    # Extract std reward
    std_reward_match = re.search(
        r"Std\s+of\s+Reward:\s*([+-]?\d*\.?\d+)", line, re.IGNORECASE
    )
    if std_reward_match:
        metrics["std_reward"] = float(std_reward_match.group(1))

    mean_group_reward_match = re.search(
        r"Mean Group Reward:\s*([+-]?\d*\.?\d+)", line, re.IGNORECASE
    )
    if mean_group_reward_match:
        metrics["mean_group_reward"] = float(mean_group_reward_match.group(1))

    time_elapsed_match = re.search(
        r"Time Elapsed:\s*([+-]?\d*\.?\d+)", line, re.IGNORECASE
    )
    if time_elapsed_match:
        metrics["time_elapsed"] = float(time_elapsed_match.group(1))

    return metrics


def main():
    """
    Main function to get user input, run mlagents-learn command,
    extract data and calculate statistics, store data in CSV table.
    Usage:
        generate-data <config_file_path> <run_id_name>
    """

    # Parse command-line arguments
    # Handle both "python script.py generate-data <config> <run_id>"
    # and "python script.py <config> <run_id>"
    args = sys.argv[1:]

    if len(args) != 3:
        print(
            "Usage: python data_extraction_script.py"
            " <config_file_path> <env_path> <run_id_name>"
        )
        print(
            "Ex: python ./data-extraction-scripts/data_extraction_script.py "
            "config/custom/SoccerTwosCustomConfigRun1.yaml "
            "training-envs/SoccerTwos_mac_env SCTWRUN1"
        )
        sys.exit(1)

    config_file_path = args[0]
    env_path = args[1]
    run_id = args[2]

    # Validate that the config file exists
    if not os.path.exists(config_file_path):
        print(f"Error: Config file not found at '{config_file_path}'")
        sys.exit(1)

    if not run_id:
        print("Error: Run-id name cannot be empty")
        sys.exit(1)

    # Extract game name from config file path
    config_filename = os.path.basename(config_file_path)
    choosen_game = os.path.splitext(config_filename)[0]

    # Extract hyperparams
    config_hyperparams = parse_config_file(config_file_path)

    # Create CSV table in training-data folder
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    training_data_dir = os.path.join(project_root, "training-data")

    # Ensure training-data directory exists
    os.makedirs(training_data_dir, exist_ok=True)

    csv_file_name = "training_data_" + run_id + ".csv"
    csv_output_path = os.path.join(training_data_dir, csv_file_name)

    if not create_csv_table(csv_output_path):
        sys.exit(1)

    # Run the training command and process output
    try:
        exit_code = run_training(
            config_file_path,
            run_id,
            env_path,
            csv_output_path,
            config_hyperparams,
        )
        if exit_code != 0:
            sys.exit(exit_code)
    except FileNotFoundError:
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

    tb_results_root = os.path.join(project_root, "results")
    tb_run_dir = os.path.join(tb_results_root, run_id, choosen_game)

    if os.path.exists(tb_run_dir):
        tb_metrics = TensorBoardMetrics(
            events_path=tb_run_dir, train_csv_path=csv_output_path
        )
        tb_metrics.append_metrics()
    else:
        print(f"Warning: Failed to access TensorBoard metrics, {tb_run_dir}")


if __name__ == "__main__":
    main()
