import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Define thread colors and markers (inspired by the provided example)
# Extended to cover up to 128 threads
thread_colors = {
    1: 'cyan', 2: 'green', 4: 'magenta', 8: 'orange',
    16: 'blue', 32: 'red', 64: 'purple', 128: 'brown'
    # Add more if needed, or handle defaults
}

thread_markers = {
    1: 'o', 2: 's', 4: '^', 8: 'D',
    16: 'X', 32: '*', 64: 'p', 128: 'h' # p=pentagon, h=hexagon
    # Add more if needed, or handle defaults
}

thread_markersize = {
    1: 12,   #make the circle marker noticeably bigger
    2: 9,
    4: 8,
    8: 8,
    16: 8,
    32: 8
}

'''def parse_perf_data(input_filename, output_filename):
    """
    Parses the perf stat output file and converts it to a CSV file.

    Args:
        input_filename (str): Path to the input .data file.
        output_filename (str): Path to the output .csv file.
    """

    # Regex to extract command, input size, and threads
    # Assumes input size and threads are the last two space-separated args
    # Adjust if the command structure is different
    command_regex = re.compile(r"Performance counter stats for '(?P<command>.*)\s+(?P<input_size>\d+)\s+(?P<threads>\d+)\s*':")

    # Regex patterns for individual metric lines
    # Making value capture robust for commas and decimals
    patterns = {
        "task_clock_msec": re.compile(r"^\s*([\d,\.]+)\s+msec\s+task-clock"),
        "context_switches": re.compile(r"^\s*([\d,]+)\s+context-switches"),
        "cpu_migrations": re.compile(r"^\s*([\d,]+)\s+cpu-migrations"),
        "page_faults": re.compile(r"^\s*([\d,]+)\s+page-faults"),
        "cycles": re.compile(r"^\s*([\d,]+)\s+cycles"),
        "instructions": re.compile(r"^\s*([\d,]+)\s+instructions"),
        "branches": re.compile(r"^\s*([\d,]+)\s+branches"),
        "branch_misses": re.compile(r"^\s*([\d,]+)\s+branch-misses"),
        "time_elapsed_sec": re.compile(r"^\s*([\d\.]+)\s+seconds time elapsed"),
        "user_time_sec": re.compile(r"^\s*([\d\.]+)\s+seconds user"),
        "sys_time_sec": re.compile(r"^\s*([\d\.]+)\s+seconds sys"),
    }

    # Define the header for the CSV file
    # Ensure order matches the patterns dictionary keys + parameters
    header = [
        "InputSize",
        "Threads",
        "task_clock_msec",
        "context_switches",
        "cpu_migrations",
        "page_faults",
        "cycles",
        "instructions",
        "branches",
        "branch_misses",
        "time_elapsed_sec",
        "user_time_sec",
        "sys_time_sec",
    ]

    current_run_data = {}
    all_runs_data = []

    try:
        with open(input_filename, 'r') as infile:
            for line in infile:
                line = line.strip()

                # Check if it's the start of a new stats block
                match_command = command_regex.search(line)
                if match_command:
                    # If we have data from a previous run, store it
                    # (unless it's the very first run)
                    if current_run_data:
                         all_runs_data.append(current_run_data)

                    # Start data for the new run
                    params = match_command.groupdict()
                    current_run_data = {
                        "InputSize": int(params['input_size']),
                        "Threads": int(params['threads']),
                    }
                    # Initialize metric fields to None or empty string
                    for key in patterns.keys():
                        current_run_data[key] = None # Or "" if you prefer
                    continue # Move to the next line

                # If we are inside a run block, try parsing metrics
                if current_run_data: # Ensure we've matched a command line first
                    for key, pattern in patterns.items():
                        match_metric = pattern.match(line)
                        if match_metric:
                            value_str = match_metric.group(1).replace(',', '') # Remove commas
                            try:
                                # Convert to float if it has a decimal, else int
                                value = float(value_str) if '.' in value_str else int(value_str)
                                current_run_data[key] = value
                            except ValueError:
                                print(f"Warning: Could not convert value '{value_str}' for key '{key}' in line: {line}", file=sys.stderr)
                                current_run_data[key] = value_str # Store as string if conversion fails
                            break # Found a match for this line, move to next line

            # Append the data from the very last run in the file
            if current_run_data:
                all_runs_data.append(current_run_data)

    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.", file=sys.stderr)
        return
    except Exception as e:
        print(f"An error occurred during parsing: {e}", file=sys.stderr)
        return

    # Write the collected data to the CSV file
    if not all_runs_data:
        print("Warning: No data parsed from the input file.", file=sys.stderr)
        return

    try:
        with open(output_filename, 'w', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=header)
            writer.writeheader()
            # Filter data to only include keys present in the header, handling missing keys gracefully
            for run_data in all_runs_data:
                 filtered_data = {key: run_data.get(key, "") for key in header} # Use get() for safety
                 writer.writerow(filtered_data)
        print(f"Successfully converted '{input_filename}' to '{output_filename}'")

    except IOError as e:
        print(f"Error writing to output file '{output_filename}': {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during CSV writing: {e}", file=sys.stderr)

def csv_to_graph(csv_filename):

    # --- Load Data ---
    df = pd.read_csv(csv_filename)

    # Clean up column names (remove leading/trailing spaces if any)
    df.columns = df.columns.str.strip()

    # --- Group and Aggregate Data ---
    # Group by InputSize and Threads, then calculate the mean for relevant columns
    grouped_data = df.groupby(['InputSize', 'Threads'])[['cycles', 'instructions', 'page_faults']].mean()

    # Calculate Instructions Per Cycle (IPC)
    # Handle potential division by zero, replace resulting NaN/inf with 0
    grouped_data['ipc'] = (grouped_data['instructions'] / grouped_data['cycles']).fillna(0)
    grouped_data['ipc'] = grouped_data['ipc'].replace([float('inf'), -float('inf')], 0)

    # Reset index to make 'InputSize' and 'Threads' regular columns for plotting
    plot_data = grouped_data.reset_index()

    # Get unique thread counts for plotting lines
    thread_counts = sorted(plot_data['Threads'].unique())

    # --- Plotting ---

    # 1. Instructions Per Cycle (IPC) Plot
    plt.figure(figsize=(10, 6)) # Set figure size for better readability
    for threads in thread_counts:
        subset = plot_data[plot_data['Threads'] == threads]
        # Sort by InputSize to ensure lines are drawn correctly
        subset = subset.sort_values('InputSize')
        plt.plot(subset['InputSize'], subset['ipc'], marker='o', linestyle='-', label=f'{threads} Threads')

    plt.xlabel('Input Size')
    plt.ylabel('Instructions Per Cycle (IPC)')
    plt.title('IPC vs. Input Size by Thread Count')
    plt.legend()
    plt.grid(True)
    plt.xscale('log') # Use log scale for x-axis if input sizes vary greatly
    plt.xticks(plot_data['InputSize'].unique(), plot_data['InputSize'].unique()) # Show actual input sizes as ticks


    # 2. Page Faults Plot
    plt.figure(figsize=(10, 6)) # Set figure size
    for threads in thread_counts:
        subset = plot_data[plot_data['Threads'] == threads]
        # Sort by InputSize
        subset = subset.sort_values('InputSize')
        plt.plot(subset['InputSize'], subset['page_faults'], marker='o', linestyle='-', label=f'{threads} Threads')

    plt.xlabel('Input Size')
    plt.ylabel('Average Page Faults')
    plt.title('Page Faults vs. Input Size by Thread Count')
    plt.legend()
    plt.grid(True)
    plt.xscale('log') # Use log scale for x-axis if input sizes vary greatly
    plt.xticks(plot_data['InputSize'].unique(), plot_data['InputSize'].unique()) # Show actual input sizes as ticks


    # --- Show Plots ---
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show()

    print("Processing complete. Plots should be displayed.")
    # Optional: print the aggregated data
    # print("\nAggregated Data:")
    # print(plot_data)'''

    

# --- How to use ---
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <input_perf.data> <output_data.csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    #df = pd.DataFrame(input_file)
    output_file = sys.argv[2]

    #parse_perf_data(input_file, output_file)