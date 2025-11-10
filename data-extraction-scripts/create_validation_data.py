#!/usr/bin/env python3
"""
Script to remove data from specified columns in a CSV file.
Keeps the column headers but removes all data in those columns.
"""

import sys
import os
import csv
import argparse


def remove_column_data(csv_file_path, columns_to_remove):
    """
    Removes data from specified columns in a CSV file while keeping the headers.
    
    Args:
        csv_file_path (str): Path to the input CSV file.
        columns_to_remove (list): List of column names to remove data from.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    # Validate that the CSV file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at '{csv_file_path}'")
        return False
    
    # Read the CSV file
    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as infile:
            # Detect delimiter
            sample = infile.read(1024)
            infile.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            
            reader = csv.DictReader(infile, delimiter=delimiter)
            fieldnames = reader.fieldnames
            
            if not fieldnames:
                print("Error: CSV file appears to be empty or has no headers")
                return False
            
            # Validate that all specified columns exist
            missing_columns = [col for col in columns_to_remove if col not in fieldnames]
            if missing_columns:
                print(f"Error: The following columns were not found in the CSV: {', '.join(missing_columns)}")
                print(f"Available columns: {', '.join(fieldnames)}")
                return False
            
            # Read all rows
            rows = []
            for row in reader:
                # Remove data from specified columns
                for col in columns_to_remove:
                    row[col] = ''
                rows.append(row)
        
        # Write the modified data back to the file
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"Successfully removed data from columns: {', '.join(columns_to_remove)}")
        print(f"Modified file: {csv_file_path}")
        return True
        
    except csv.Error as e:
        print(f"Error reading/writing CSV file: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """
    Main function to parse command-line arguments and remove column data.
    
    Usage:
        create-validation-data <csv_file_path> <column1> [column2] [column3] ...
    """
    # Parse command-line arguments
    # Handle both "python script.py create-validation-data <csv> <columns>" and "python script.py <csv> <columns>"
    args = sys.argv[1:]
    
    # If first arg is "create-validation-data", skip it
    if len(args) > 0 and args[0] == "create-validation-data":
        args = args[1:]
    
    if len(args) < 2:
        print("Usage: create-validation-data <csv_file_path> <column1> [column2] [column3] ...")
        print("Example: create-validation-data data.csv column1 column2")
        print("Example: create-validation-data training_data.csv mean_reward std_reward")
        sys.exit(1)
    
    csv_file_path = args[0]
    columns_to_remove = args[1:]
    
    # Validate CSV file path
    if not csv_file_path:
        print("Error: CSV file path cannot be empty")
        sys.exit(1)
    
    # Validate columns
    if not columns_to_remove:
        print("Error: At least one column name must be specified")
        sys.exit(1)
    
    # Remove data from specified columns
    success = remove_column_data(csv_file_path, columns_to_remove)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

