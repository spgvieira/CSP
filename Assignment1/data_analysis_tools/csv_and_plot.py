#This script takes our perf and wall time textfiles, outputting a combined csv 
#AND two plot graph pngs - one comparing hashbits with perf_vals, and another comparing wall_time with hashbits.
#both drawing lines/plots for 1, 2, 4, 8, 16 and up to 32 threads.
import argparse
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)

#change or add folders here you'd like plots to be written to.
csv_results_folder = os.path.join(project_dir, 'csv')
plots_results_folder = os.path.join(project_dir, 'plots')

#define mapping of thread counts to colors (used in both plots)
thread_colors = {
    1: 'cyan',
    2: 'green',
    4: 'magenta',
    8: 'orange',
    16: 'blue',
    32: 'red'
}


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
            parts = line.strip().split(',')
            threads = int(parts[0])
            hash_bits = int(parts[1])
            time = float(parts[2])
            data[(threads, hash_bits)].append(time)

    csv_rows = []
    for (threads, hash_bits), times in data.items():
        avg_time = sum(times) / len(times)
        csv_rows.append([threads, hash_bits, round(avg_time, 6)])

    return csv_rows

def convert_merge_to_csv(perf_results, time_results):
    #create a dictionary using time_results for easy lookup when merging with perf_results
    time_dict = {}
    for t, h, time in time_results:
        time_dict[(t, h)] = time

    #merge wall_time into each entry from perf_results
    for entry in perf_results:
        threads = int(entry['threads'])
        hashbits = int(entry['hashbits'])
        entry['wall_time'] = time_dict.get((threads, hashbits), None)

    #determine the headers for the csv
    headers = []
    if perf_results:
        headers = ['threads', 'hashbits']
        other_headers = [key for key in perf_results[0].keys() 
                         if key not in headers and key not in ('time_elapsed', 'wall_time')]
        headers += other_headers
        headers += ['time_elapsed', 'wall_time']
    else:
        headers = []

    #create a dataframe (df) for future use
    df = pd.DataFrame(perf_results, columns=headers)
    return df

def plot_hashbits_versus_perf_vals(merged_df, date, programName): 
    csvData = merged_df
    #convert columns for plotting clarity and convert cpu-cycles to billions and cache-misses to millions
    csvData['cpu-cycles (B)'] = csvData['cpu-cycles'] / 1e9
    csvData['cache-misses (M)'] = csvData['cache-misses'] / 1e6
    csvData['dTLB-load-misses (M)'] = csvData['dTLB-load-misses'] / 1e6
    csvData['page-faults (M)'] = csvData['page-faults'] / 1e6

    #list of metrics to plot and corresponding labels
    metrics = [
        ('cpu-cycles (B)', 'CPU Cycles (Billions)'),
        ('cache-misses (M)', 'Cache Misses (Millions)'),
        ('page-faults (M)', 'Page Faults (Millions)'),
        ('cpu-migrations', 'CPU Migrations'),
        ('dTLB-load-misses (M)', 'dTLB Load Misses (Millions)'),
        ('context-switches', 'Context Switches')
    ]

    #create subplots: 2 rows x 3 columns
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    axes = axes.flatten()

    #loop through each metric and subplot axis
    for ax, (col, ylabel) in zip(axes, metrics):
        for thread in sorted(merged_df['threads'].unique()):
            color = thread_colors.get(thread, 'black')#defaults to black if no color is set in dictionary top of file.
            subset = merged_df[merged_df['threads'] == thread]
            #use plot with marker "o" and line style "-" to connect dots.
            ax.plot(subset['hashbits'], subset[col],
                    marker='o', markersize=4, linestyle='-', linewidth=1,
                    label=f'{thread} thread', color=color)
        ax.set_title(ylabel)
        ax.set_xlabel('Hashbits')
        ax.set_ylabel(ylabel)
        ax.legend()
        #ensure x-axis ticks are the integers 1 through 18 (for hashbits 1-18)
        ax.set_xticks(range(1, 19))

    plt.tight_layout()
    output_path = os.path.join(plots_results_folder, f'{date}_{programName}_hashbits_vs_perfvals.png')
    plt.savefig(output_path, dpi=300)


def plot_tuples_persec_hashbits(merged_df, date, programName):
    csvData = merged_df
    csvData['throughput_mtps'] = 16.777216 / csvData['wall_time']

    #here we filter the stuff we want from csv
    sel_threads = [1, 2, 4, 8, 16, 32]
    csvData = csvData[csvData['threads'].isin(sel_threads)].sort_values(['threads', 'hashbits'])

    #creating the plot
    plt.figure(figsize=(12,7))

    #plot a line for each thread count 1-32:
    for threads in sel_threads:
        subset = csvData[csvData['threads'] == threads]
        if not subset.empty:
            color = thread_colors.get(threads, 'black')#defaults to black if no color is set in dictionary top of file.
            plt.plot(subset['hashbits'],
                    subset['throughput_mtps'],#change this to "time_elapsed" if you want to track perf time
                    marker='o',
                    linestyle='-',
                    linewidth=2,
                    markersize=8,
                    label=f' {threads} Threads',
                    color=color)
            
    #appearance of the plot graph
    plt.xlabel('hashbits', fontsize=12)
    plt.ylabel('Milions of tuples per second', fontsize = 12)
    plt.title('Millions of tuples per second vs Hashbits by thread count.', fontsize = 14)
    plt.xticks(range(1, 19)) #hashbits from 1 to 18
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.legend(title='thread Count', fontsize=10)
    plt.tight_layout()

    output_path = os.path.join(plots_results_folder, f'{date}_{programName}_performance_plot.png')
    #save plot
    plt.savefig(output_path, dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse performance results and wall times into CSV')
    parser.add_argument('perf_file', help='Input text file for performance results (e.g., perf_results.txt)')
    parser.add_argument('time_file', help='Input text file for wall times (e.g., time_results.txt)')
    parser.add_argument('--output_csv', default='output.csv',
                        help='Default output CSV file name (overridden by file name rules)')
    args = parser.parse_args()
    #parse the two input files
    perf_results = parse_results(args.perf_file)
    time_results = parse_time(args.time_file)

    #merge the results into a DataFrame
    merged_df = convert_merge_to_csv(perf_results, time_results)

    #need to declare these values as type int, otherwise plot functions will read them in the dataframe as strings, 
    #and it will mess up when it needs to sort by threads/hashbits and iterate over them to plot.
    #(wasn't a problem before because we converted straight to raw csv, which already did the int conversion automatically.)
    merged_df['threads'] = merged_df['threads'].astype(int)
    merged_df['hashbits'] = merged_df['hashbits'].astype(int)

    #output_folder = "csv"  #change to desired folder

    #determine output CSV file name based on the perf_file prefix:
    perf_filename = os.path.basename(args.perf_file)
    if perf_filename.startswith("indep"):#independent
        prefix = "indep"
    elif perf_filename.startswith("conc"):#concurrent
        prefix = "conc"
    else:
        prefix = "output"#default if none of them are present.

    today_str = datetime.today().strftime('%d_%m')
    output_filename = f"output_{prefix}_{today_str}.csv"

    #the path to where the output csv is saved to.
    output_csv_path = os.path.join(csv_results_folder, output_filename)

    #write the dataframe to CSV
    merged_df.to_csv(output_csv_path, index=False)

    plot_hashbits_versus_perf_vals(merged_df, today_str, prefix)
    plot_tuples_persec_hashbits(merged_df, today_str,prefix)



#def plotting (results_perf, results_time):


    