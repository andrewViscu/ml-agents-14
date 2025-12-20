#!/usr/bin/env python3
import subprocess
import time
from csv_table_creator import write_csv_row, get_csv_columns
from resource_py_script import ResourceMonitor

def run_training(config_file_path, run_id, chosen_game, learning_algorithm, csv_output_path, env_path):
    """
    Runs mlagents-learn command and processes output in real-time, writing metrics to CSV.
    
    Args:
        config_file_path (str): Path to the config file.
        run_id (str): Unique identifier for the training run.
        chosen_game (str): Name of the game extracted from config file.
        learning_algorithm (str): Learning algorithm extracted from config file path.
        csv_output_path (str): Path to the CSV file where data will be written.
    
    Returns:
        int: Exit code of the mlagents-learn command (0 for success, non-zero for failure).
    
    Raises:
        FileNotFoundError: If mlagents-learn command is not found.
        KeyboardInterrupt: If the command is interrupted by user.
    """
    # Construct the command
    command = ["mlagents-learn", config_file_path, "--run-id", run_id]
    if env_path:
        command.append(f"--env={env_path}")

    command += ["--run-id", run_id]
    
    print(f"\nRunning command: {' '.join(command)}\n")
    
    # Get CSV columns for writing rows
    columns = get_csv_columns()
    
    # Run the command and capture output in real-time
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Start resource monitoring for the training process
        resource_monitor = ResourceMonitor(ml_pid=process.pid, interval_s=1.0)
        resource_monitor.start()
        
        # Give the monitor a moment to start collecting data
        time.sleep(0.5)
        
        # Variables to store current metrics for a step
        current_step = None
        current_mean_reward = None
        current_std_reward = None
        current_mean_group_reward = None
        current_time_elapsed = None
        written_steps = set()  # Track steps we've already written to avoid duplicates
        
        from data_extraction_script import parse_mlagents_output
        
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
            if metrics["mean_group_reward"] is not None:
                current_mean_group_reward = metrics["mean_group_reward"]
            if metrics["time_elapsed"] is not None:
                current_time_elapsed = metrics["time_elapsed"]
            
            # If we have a step and at least one metric, and haven't written this step yet, write to CSV
            if current_step is not None and current_step not in written_steps:
                if current_mean_reward is not None or current_std_reward is not None or current_mean_group_reward is not None:

                    # Calculate performance and resource usage data (called on each line with metrics)
                    resource_usage_data = resource_monitor.calculate_resource_usage_data()

                    # Prepare row data - handle None values properly
                    def format_value(val):
                        """Convert None to empty string, otherwise return the value."""
                        return '' if val is None else val
                    
                    row_data = {
                        "step": current_step,
                        "mean_reward": current_mean_reward if current_mean_reward is not None else '',
                        "mean_group_reward": current_mean_group_reward if current_mean_group_reward is not None else '',
                        "std_reward": current_std_reward if current_std_reward is not None else '',
                        "time_elapsed": current_time_elapsed if current_time_elapsed is not None else '',
                        "memory_usage_avg_mb": format_value(resource_usage_data.get("memory usage avg mb")),
                        "memory_usage_peak_mb": format_value(resource_usage_data.get("memory usage peak mb")),
                        "cpu_usage_avg_mb": format_value(resource_usage_data.get("cpu usage avg percent")),
                        "cpu_usage_peak_mb": format_value(resource_usage_data.get("cpu usage peak percent")),
                        "gpu_usage_avg_mb": format_value(resource_usage_data.get("gpu usage avg mb")),
                        "gpu_usage_peak_mb": format_value(resource_usage_data.get("gpu usage peak mb")),
                    }
                    
                    # Write to CSV
                    write_csv_row(csv_output_path, row_data, columns)
                    written_steps.add(current_step)
        
        # Wait for process to complete
        process.wait()
        
        # Stop resource monitoring
        resource_monitor.stop()
        resource_monitor.join(timeout=2.0)  # Wait up to 2 seconds for thread to finish
        
        if process.returncode == 0:
            print("\n\nCommand completed successfully!")
            print(f"Data saved to '{csv_output_path}'")
        else:
            print(f"\n\nCommand completed with exit code {process.returncode}")
        
        return process.returncode
            
    except FileNotFoundError:
        print("\nError: 'mlagents-learn' command not found. Make sure ML-Agents is installed and in your PATH.")
        raise
    except KeyboardInterrupt:
        print("\n\nCommand interrupted by user.")
        if 'resource_monitor' in locals():
            resource_monitor.stop()
            resource_monitor.join(timeout=2.0)
        if process:
            process.terminate()
        raise
    except Exception as e:
        print(f"\nError: {e}")
        raise

