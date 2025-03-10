import csv
import argparse
from collections import defaultdict
import sys
import pandas as pd
import matplotlib.pyplot as plt

def parse_results(file_path):
    entries = []
    current_entry = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('Performance counter stats for'):
                # Start new entry
                if current_entry is not None:
                    entries.append(current_entry)
                current_entry = {}
                # Extract command parameters (threads and hashbits)
                command = line.split("'")[1]
                parts = command.split()
                current_entry['threads'] = parts[1]
                current_entry['hashbits'] = parts[2]
            elif line.startswith('#'):
                continue  # Skip comment lines with dates
            else:
                parts = line.split()
                if not parts:
                    continue
                # Check for time elapsed line
                if 'seconds time elapsed' in line:
                    current_entry['time_elapsed'] = float(parts[0])
                    entries.append(current_entry)
                    current_entry = None
                else:
                    # Check if the line starts with a numeric value (metric line)
                    first_part = parts[0].replace(',', '')
                    if first_part.isdigit():
                        metric_value = int(first_part)
                        metric_name = parts[1]
                        current_entry[metric_name] = metric_value
    return entries

def parse_time(file_path):

    data = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            threads = int(parts[2])
            hash_bits = int(parts[6])
            time = float(parts[7].split(':')[1])
            data[(threads, hash_bits)].append(time)

    csv_rows = []
    for (threads, hash_bits), times in data.items():
        avg_time = sum(times) / len(times)
        csv_rows.append([threads, hash_bits, round(avg_time, 6)])

    return csv_rows

def convert_to_CSV (perf_results, time_results):
    # Create a dictionary from time_results for easy lookup
    time_dict = {}
    for t, h, time in time_results:
        time_dict[(t, h)] = time

    # Merge wall_time into each entry in perf_results
    for entry in perf_results:
        # Convert threads and hashbits to integers for lookup
        threads = int(entry['threads'])
        hashbits = int(entry['hashbits'])
        entry['wall_time'] = time_dict.get((threads, hashbits), None)

    # Determine CSV headers
    headers = []
    if perf_results:
        headers = ['threads', 'hashbits']
        # Collect other headers (excluding time_elapsed and wall_time)
        other_headers = []
        for key in perf_results[0].keys():
            if key not in headers and key not in ('time_elapsed', 'wall_time'):
                other_headers.append(key)
        headers += other_headers
        headers += ['time_elapsed', 'wall_time']  # Ensure these are last
    else:
        headers = []

    # Write to CSV
    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for entry in perf_results:
            # Ensure each entry has all headers, filling missing ones with empty strings
            row = {header: entry.get(header, '') for header in headers}
            writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse performance results and wall times into CSV')
    parser.add_argument('perf_file', help='Input text file for performance results (e.g., perf_results.txt)')
    parser.add_argument('time_file', help='Input text file for wall times (e.g., time_results.txt)')
    parser.add_argument('-o', '--output', default='output.csv',
                       help='Output CSV file (default: output.csv)')
    args = parser.parse_args()

    # Parse the two input files
    perf_results = parse_results(args.perf_file)
    time_results = parse_time(args.time_file)

    # Convert and merge the results into a CSV
    convert_to_CSV(perf_results, time_results)

    print(f"Results have been saved to {args.output}")
#def plotting (results_perf, results_time):
    