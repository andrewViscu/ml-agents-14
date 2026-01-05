#!/usr/bin/env python3
import csv
import os


def get_csv_columns():
    """
    Get the list of CSV column headers.

    Returns:
        list: List of column names in order.
    """
    return [
        "step",
        "mean_reward",
        "mean_group_reward",
        "std_reward",
        "time_elapsed",
        "memory_usage_avg_mb",
        "memory_usage_peak_mb",
        "cpu_usage_avg_percent",
        "cpu_usage_peak_percent",
        "gpu_usage_avg_mb",
        "gpu_usage_peak_mb",
        "trainer_type",
        "batch_size",
        "buffer_size",
        "learning_rate",
        "beta",
        "epsilon",
        "lambd",
        "num_epoch",
        "learning_rate_schedule",
    ]


def create_csv_table(output_path="training_data.csv"):
    """
    Create a CSV table with predefined columns.

    Args:
        output_path (str): Path where the CSV file will be created. Defaults to
        "training_data.csv".

    Returns:
        bool: True if CSV was created successfully, False otherwise.
    """
    columns = get_csv_columns()

    # Check if file already exists
    if os.path.exists(output_path):
        response = (
            input(f"File '{output_path}' already exists. Overwrite? (y/n): ")
            .strip()
            .lower()
        )
        if response != "y":
            print("Operation cancelled.")
            return False

    # Create the CSV file with headers
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)

        print(f"Successfully created CSV table at '{output_path}'")
        return True

    except Exception as e:
        print(f"Error creating CSV file: {e}")
        return False


def write_csv_row(output_path, row_data, columns):
    """
    Write a row of data to the CSV file.

    Args:
        output_path (str): Path to the CSV file.
        row_data (dict): Dictionary containing the row data.
        columns (list): List of column names in order.
    """
    try:
        with open(output_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            row = [row_data.get(col, "") for col in columns]
            writer.writerow(row)
    except Exception as e:
        print(f"Error writing to CSV file: {e}")


def main():
    """
    Main function to create the CSV table.
    """
    # Optionally get output path from user, or use default
    output_path = input(
        "Enter output CSV file path (press Enter for 'training_data.csv'): "
    ).strip()

    if not output_path:
        output_path = "training_data.csv"

    create_csv_table(output_path)


if __name__ == "__main__":
    main()
