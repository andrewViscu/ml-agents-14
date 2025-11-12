#!/usr/bin/env python3
import sys
import os
import re
from datetime import datetime
from csv_table_creator import create_csv_table
from training_runner import run_training


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
    
    # Create CSV table in training-data folder
    # Get the project root directory (one level up from data-extraction-scripts)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    training_data_dir = os.path.join(project_root, "training-data")
    
    # Ensure training-data directory exists
    os.makedirs(training_data_dir, exist_ok=True)
    
    csv_output_path = os.path.join(training_data_dir, "training_data.csv")
    
    if not create_csv_table(csv_output_path):
        sys.exit(1)
    
    # Run the training command and process output
    try:
        exit_code = run_training(config_file_path, run_id, chosen_game, learning_algorithm, csv_output_path)
        if exit_code != 0:
            sys.exit(exit_code)
    except FileNotFoundError:
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

