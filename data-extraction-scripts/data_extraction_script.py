#!/usr/bin/env python3
import subprocess
import sys
import os
import csv
import re
from datetime import datetime
from csv_table_creator import create_csv_table, write_csv_row, get_csv_columns

# ----------- Mikulas's Code -----------

# ----------- Louis's Code -----------

# ----------- David's Code -----------

def get_timestamp():
    """
    Records exact time of training.
    
    Returns:
        str: Timestamp of the training session.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_run_id(run_id_name):
    """
    Gets unique identifier for each training run to track specific runs and
    reference during analysis.
    
    Args:
        run_id_name (str): The run ID name provided by the user.
    
    Returns:
        str: Unique run identifier.
    """
    return run_id_name


def get_environment(chosen_game):
    """
    Identifies the unity game environment to compare algorithm performance
    across different tasks.
    
    Args:
        chosen_game (str): The name of the game extracted from config file.
    
    Returns:
        str: Environment identifier.
    """
    
    single_agent_games = ["Basic", "3DBall", "3DBallHard", "3DBall_randomize", "GridWorld", "PushBlock", "WallJump", "Crawler", "Hallway"]
    multi_agent_games = ["SoccerTwos", "Tennis", "BouncyBalls"]
    
    if chosen_game in single_agent_games:
        return "single-agent"
    elif chosen_game in multi_agent_games:
        return "multi-agent"
    else:
        return "unknown"


def get_algorithm(learning_algorithm):
    """
    Identifies the reinforcement learning algorithm used.
    
    Returns:
        str: Algorithm identifier.
    """

    return learning_algorithm


def get_step(step_value):
    """
    Gets training progress indicator to calculate rewards.
    
    Args:
        step_value (int): The step value extracted from mlagents-learn output.
    
    Returns:
        int: Current training step.
    """
    return step_value


def calculate_general_data(chosen_game, learning_algorithm, run_id_name, step_value):
    """
    Calculates general data metrics for training session.
    
    Args:
        chosen_game (str): The name of the game extracted from config file.
        learning_algorithm (str): The learning algorithm extracted from config file path.
        run_id_name (str): The run ID name provided by the user.
        step_value (int): The step value extracted from mlagents-learn output.
    
    Returns:
        dict: Dictionary containing general data metrics:
            - timestamp: Exact time of training
            - run_id: Unique identifier for training run
            - environment: Unity game environment identifier
            - algorithm: Reinforcement learning algorithm used
            - step: Training progress indicator
    """
    timestamp = get_timestamp()
    run_id = get_run_id(run_id_name)
    environment = get_environment(chosen_game)
    algorithm = get_algorithm(learning_algorithm)
    step = get_step(step_value)
    
    return {
        "timestamp": timestamp,
        "run_id": run_id,
        "environment": environment,
        "algorithm": algorithm,
        "step": step
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
            - policy_loss: Policy loss (float or None)
    """
    metrics = {
        "step": None,
        "mean_reward": None,
        "std_reward": None,
        "policy_loss": None
    }
    
    # Extract step: "Step: 1000" or similar
    step_match = re.search(r'Step:\s*(\d+)', line, re.IGNORECASE)
    if step_match:
        metrics["step"] = int(step_match.group(1))
    
    # Extract mean reward: "Mean Reward: 1.234" or similar
    mean_reward_match = re.search(r'Mean Reward:\s*([+-]?\d*\.?\d+)', line, re.IGNORECASE)
    if mean_reward_match:
        metrics["mean_reward"] = float(mean_reward_match.group(1))
    
    # Extract std reward: "Std of Reward: 0.123" or "Std of Reward: 0.123" or similar
    std_reward_match = re.search(r'Std\s+of\s+Reward:\s*([+-]?\d*\.?\d+)', line, re.IGNORECASE)
    if std_reward_match:
        metrics["std_reward"] = float(std_reward_match.group(1))
    
    # Extract policy loss: "Losses/Policy Loss: 0.123" or "Policy Loss: 0.123" or similar
    policy_loss_match = re.search(r'(?:Losses/)?Policy\s+Loss[:\s]+([+-]?\d*\.?\d+)', line, re.IGNORECASE)
    if policy_loss_match:
        metrics["policy_loss"] = float(policy_loss_match.group(1))
    
    return metrics


# ----------- Main Function -----------
def main():
    """
    Main function to get user input, run mlagents-learn command, extract data and calculate statistics, store data in CSV table.
    
    Usage:
        generate-data <config_file_path> <run_id_name>
    """
    
    # Parse command-line arguments
    # Handle both "python script.py generate-data <config> <run_id>" and "python script.py <config> <run_id>"
    args = sys.argv[1:]
    
    # If first arg is "generate-data", skip it
    if len(args) > 0 and args[0] == "generate-data":
        args = args[1:]
    
    if len(args) < 2:
        print("Usage: generate-data <config_file_path> <run_id_name>")
        print("Example: generate-data config/ppo/3DBall.yaml my_run_001")
        sys.exit(1)
    
    config_file_path = args[0]
    run_id = args[1]
    
    # Validate that the config file exists
    if not os.path.exists(config_file_path):
        print(f"Error: Config file not found at '{config_file_path}'")
        sys.exit(1)
    
    if not run_id:
        print("Error: Run-id name cannot be empty")
        sys.exit(1)
    
    # Extract game name from config file path
    config_filename = os.path.basename(config_file_path)
    chosen_game = os.path.splitext(config_filename)[0]
    
    # Extract learning algorithm from config file path (second part of path)
    # Example: config/ppo/3DBall.yaml -> learning_algorithm = "ppo"
    path_parts = os.path.normpath(config_file_path).split(os.sep)
    learning_algorithm = path_parts[1] if len(path_parts) > 1 else None
    
    # Create CSV table
    csv_output_path = "training_data.csv"
    columns = get_csv_columns()
    
    if not create_csv_table(csv_output_path):
        sys.exit(1)
    
    # Construct the command
    command = ["mlagents-learn", config_file_path, "--run-id", run_id]
    
    print(f"\nRunning command: {' '.join(command)}\n")
    
    # Run the command and capture output in real-time
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Variables to store current metrics for a step
        current_step = None
        current_mean_reward = None
        current_std_reward = None
        current_policy_loss = None
        written_steps = set()  # Track steps we've already written to avoid duplicates
        
        # Read output line by line
        for line in process.stdout:
            # Print the line to console
            print(line, end='')
            
            # Parse the line for metrics
            metrics = parse_mlagents_output(line)
            
            # Update current step if found
            if metrics["step"] is not None:
                current_step = metrics["step"]
            
            # Update current metrics if found
            if metrics["mean_reward"] is not None:
                current_mean_reward = metrics["mean_reward"]
            if metrics["std_reward"] is not None:
                current_std_reward = metrics["std_reward"]
            if metrics["policy_loss"] is not None:
                current_policy_loss = metrics["policy_loss"]
            
            # If we have a step and at least one metric, and haven't written this step yet, write to CSV
            if current_step is not None and current_step not in written_steps:
                if current_mean_reward is not None or current_std_reward is not None or current_policy_loss is not None:
                    # Calculate general data (called on each line with metrics)
                    general_data = calculate_general_data(chosen_game, learning_algorithm, run_id, current_step)
                    
                    # Prepare row data
                    row_data = {
                        "step": current_step,
                        "mean_reward": current_mean_reward if current_mean_reward is not None else '',
                        "std_reward": current_std_reward if current_std_reward is not None else '',
                        "policy_loss": current_policy_loss if current_policy_loss is not None else '',
                        "timestamp": general_data["timestamp"],
                        "run_id": general_data["run_id"],
                        "environment": general_data["environment"],
                        "algorithm": general_data["algorithm"],
                        "success_rates_percentage": '',  # TODO: Implement when available
                        "cumulative_reward": '',  # TODO: Implement when available
                        "steps_per_episode": '',  # TODO: Implement when available
                        "training_time": '',  # TODO: Implement when available
                        "memory_usage_avg_mb": '',  # TODO: Implement when available
                        "memory_usage_peak_mb": '',  # TODO: Implement when available
                        "cpu_usage_avg_mb": '',  # TODO: Implement when available
                        "cpu_usage_peak_mb": '',  # TODO: Implement when available
                        "gpu_usage_avg_mb": '',  # TODO: Implement when available
                        "gpu_usage_peak_mb": '',  # TODO: Implement when available
                        "entropy": '',  # TODO: Implement when available
                        "value_loss": ''  # TODO: Implement when available
                    }
                    
                    # Write to CSV
                    write_csv_row(csv_output_path, row_data, columns)
                    written_steps.add(current_step)
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode == 0:
            print("\n\nCommand completed successfully!")
            print(f"Data saved to '{csv_output_path}'")
        else:
            print(f"\n\nCommand completed with exit code {process.returncode}")
            sys.exit(process.returncode)
            
    except FileNotFoundError:
        print("\nError: 'mlagents-learn' command not found. Make sure ML-Agents is installed and in your PATH.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nCommand interrupted by user.")
        if process:
            process.terminate()
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

