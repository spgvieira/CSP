import csv

def parse_results(file_path, output_csv):
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

    # Determine CSV headers based on the first entry's keys
    if entries:
        headers = ['threads', 'hashbits']
        # Preserve the order of metrics as they appear, excluding special fields
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
    parse_results('results.txt', 'output.csv')