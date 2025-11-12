from datetime import datetime

# ----------- Game Name Lists -----------
single_agent_games = ["Basic", "3DBall", "3DBallHard", "3DBall_randomize", "GridWorld", "PushBlock", "WallJump", "Crawler", "Hallway"]
multi_agent_games = ["SoccerTwos", "Tennis", "BouncyBalls"]

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