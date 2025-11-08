#!/usr/bin/env python3
import csv
import os
from datetime import datetime


def create_csv_table(output_path="training_data.csv"):
    """
    Create an empty CSV table with predefined columns.
    
    Args:
        output_path (str): Path where the CSV file will be created. Defaults to "training_data.csv".
    """

    # Define the column headers
    columns = [
        "timestamp",
        "run_id",
        "environment",
        "algorithm",
        "step",
        "success_rates_percentage",
        "cumulative_reward",
        "steps_per_episode",
        "training_time",
        "memory_usage_avg_mb",
        "memory_usage_peak_mb",
        "cpu_usage_avg_mb",
        "cpu_usage_peak_mb",
        "gpu_usage_avg_mb",
        "gpu_usage_peak_mb",
        "entropy",
        "policy_loss",
        "value_loss"
    ]
    
    # Check if file already exists
    if os.path.exists(output_path):
        response = input(f"File '{output_path}' already exists. Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("Operation cancelled.")
            return
    
    # Create the CSV file with headers
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)
        
        print(f"Successfully created CSV table at '{output_path}'")
        print(f"Columns: {', '.join(columns)}")
        print(f"Total columns: {len(columns)}")
        
    except Exception as e:
        print(f"Error creating CSV file: {e}")


def main():
    """
    Main function to create the CSV table.
    """
    # Optionally get output path from user, or use default
    output_path = input("Enter output CSV file path (press Enter for 'training_data.csv'): ").strip()
    
    if not output_path:
        output_path = "training_data.csv"
    
    create_csv_table(output_path)


if __name__ == "__main__":
    main()

