#!/usr/bin/env python3
import subprocess
import sys
import os

# ----------- Mikulas's Code -----------

# ----------- Louis's Code -----------

# ----------- David's Code -----------

def get_timestamp():
    """
    Records exact time of training.
    
    Returns:
        str: Timestamp of the training session.
    """
    # TODO: Implement timestamp extraction
    pass


def get_run_id():
    """
    Gets unique identifier for each training run to track specific runs and
    reference during analysis.
    
    Returns:
        str: Unique run identifier.
    """
    # TODO: Implement run_id extraction
    pass


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


def get_step():
    """
    Gets training progress indicator to calculate rewards.
    
    Returns:
        int: Current training step.
    """
    # TODO: Implement step extraction
    pass


def calculate_general_data(chosen_game):
    """
    Calculates general data metrics for training session.
    
    Args:
        chosen_game (str): The name of the game extracted from config file.
    
    Returns:
        dict: Dictionary containing general data metrics:
            - timestamp: Exact time of training
            - run_id: Unique identifier for training run
            - environment: Unity game environment identifier
            - algorithm: Reinforcement learning algorithm used
            - step: Training progress indicator
    """
    timestamp = get_timestamp()
    run_id = get_run_id()
    environment = get_environment(chosen_game)
    algorithm = get_algorithm(learning_algorithm)
    step = get_step()
    
    return {
        "timestamp": timestamp,
        "run_id": run_id,
        "environment": environment,
        "algorithm": algorithm,
        "step": step
    }


# ----------- Main Function -----------
def main():
    """
    Main function to get user input, run mlagents-learn command, extract data and calculate statistics, store data in CSV table.
    """
    
    # Get config file path from user
    config_file_path = input("Enter the config file path: ").strip()
    
    # Validate that the config file exists
    if not os.path.exists(config_file_path):
        print(f"Error: Config file not found at '{config_file_path}'")
        sys.exit(1)
    
    # Extract game name from config file path
    config_filename = os.path.basename(config_file_path)
    chosen_game = os.path.splitext(config_filename)[0]
    
    # Extract learning algorithm from config file path (second part of path)
    # Example: config/ppo/3DBall.yaml -> learning_algorithm = "ppo"
    path_parts = os.path.normpath(config_file_path).split(os.sep)
    learning_algorithm = path_parts[1] if len(path_parts) > 1 else None
    
    # Get run-id name from user
    run_id = input("Enter the run-id name: ").strip()
    
    if not run_id:
        print("Error: Run-id name cannot be empty")
        sys.exit(1)
    
    # Construct the command
    command = ["mlagents-learn", config_file_path, "--run-id", run_id]
    
    print(f"\nRunning command: {' '.join(command)}\n")
    
    # Run the command
    try:
        result = subprocess.run(command, check=True)
        print("\nCommand completed successfully!")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\nError: Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("\nError: 'mlagents-learn' command not found. Make sure ML-Agents is installed and in your PATH.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nCommand interrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()

