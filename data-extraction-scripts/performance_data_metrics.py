def get_success_rate():
    """
    Calculate the success rate accross a single data collection run.
    """
    # TODO: Implement with correct success rate calculation.
    
def get_cumulative_reward():
    """
    Calculate the average cumulative reward accross a single data collection run.
    """
    # TODO: Implement with correct cumulative reward calculation.

def calculate_performance_data():
    """
    Calculate performance data metrics for a single data collection run.

    Args:
        

    Returns:
        dict: Dictionary containing performance data metrics:
            - success_rate: Success rate for the data collection run.
            - cumulative_reward: Cumulative reward for the data collection run.
    """
    success_rate = get_success_rate()
    cumulative_reward = get_cumulative_reward()
    return {
        "success_rate": success_rate,
        "cumulative_reward": cumulative_reward
    }