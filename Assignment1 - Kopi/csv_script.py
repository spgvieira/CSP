import csv
import argparse
from collections import defaultdict

def parse_performance(file_path):
    """Parse the performance counter stats file"""
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

def parse_time_results(time_file):
    #Parse time results file and return


    # Determine CSV headers
    if entries:
        headers = ['threads', 'hashbits']
        first_entry_keys = list(entries[0].keys())
        for key in first_entry_keys:
            if key not in headers and key != 'time_elapsed':
                headers.append(key)
        headers.append('time_elapsed')
    else:
        headers = []

    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse performance results into CSV')
    parser.add_argument('input', help='Input text file (e.g., results.txt)')
    parser.add_argument('-o', '--output', default='output.csv',
                       help='Output CSV file (default: output.csv)')
    args = parser.parse_args()
    
    parse_results(args.input, args.output)