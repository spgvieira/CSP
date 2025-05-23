#This script takes our perf and wall time textfiles, outputting a combined csv 
#AND two plot graph pngs - one comparing hashbits with perf_vals, and another comparing wall_time with hashbits.
#both drawing lines/plots for 1, 2, 4, 8, 16 and up to 32 threads.
import argparse
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import re
import numpy as np
import math
#Code made by Su Mei Gwen Ho, Sara Vieira & Sophus Kaae Merved, with inspiration from Google's Gemini LLM

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)

#change or add folders here you'd like plots to be written to.
csv_results_folder = os.path.join(project_dir, 'csv')
plots_results_folder = os.path.join(project_dir, 'plots/18_04_graphs_symbols_no_title')

#define mapping of thread counts to colors (used in both plots)
thread_colors = {
    1: 'cyan',
    2: 'green',
    4: 'magenta',
    8: 'orange',
    16: 'blue',
    32: 'red'
}

thread_markers = {
    1:  'o',
    2:  's',
    4:  '^',
    8:  'D',
    16: 'X',
    32: '*'
}

thread_markersize = {
    1: 12,   #make the circle marker noticeably bigger
    2: 9,
    4: 8,
    8: 8,
    16: 8,
    32: 8
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

#for rounding down numbers to 10,15,20 etc.
def round_down(value, step):
    return math.floor(value / step) * step

def plot_hashbits_versus_perf_vals(merged_df, date, programName): 
    #csvData = merged_df
    csvData = merged_df
    NUM_TUPLES = 16777216

    #convert columns for plotting clarity and to make them show x per tuple.
    csvData['cpu-cycles per Tuple'] = csvData['cpu-cycles'] / NUM_TUPLES
    csvData['cache-misses per Tuple'] = csvData['cache-misses'] / NUM_TUPLES
    csvData['dTLB-load-misses per Tuple'] = csvData['dTLB-load-misses'] / NUM_TUPLES
    csvData['page-faults per Tuple'] = csvData['page-faults'] / NUM_TUPLES
    #also calculate for metrics that aren't scaled, but can be commented out later if wanted.
    #csvData['cpu-migrations per Tuple'] = csvData['cpu-migrations'] / NUM_TUPLES
    #csvData['context-switches per Tuple'] = csvData['context-switches'] / NUM_TUPLES

    #list of metrics to plot and corresponding labels
    metrics = [
        ('cpu-cycles per Tuple', 'CPU Cycles per Tuple'),
        ('cache-misses per Tuple', 'Cache Misses per Tuple'),
        ('page-faults per Tuple', 'Page Faults per Tuple'),
        ('cpu-migrations', 'CPU Migrations'),
        ('dTLB-load-misses per Tuple', 'dTLB Load Misses per Tuple'),
        ('context-switches', 'Context Switches')
    ]

        #define max y-values and y-axis tick intervals based on programName
    y_settings = {
        "indep": {
            'cpu-cycles per Tuple': (3400, 125),
            'cache-misses per Tuple': (7, 1),
            'page-faults per Tuple': (0.020, 0.002),
            'cpu-migrations': (35, 5),
            'dTLB-load-misses per Tuple': (2.2, 0.2)
            #'context-switches per Tuple': (0.001, 0.0001)
        },
        "conc": {
            'cpu-cycles per Tuple': (7500, 250),
            'cache-misses per Tuple': (7, 1),
            'page-faults per Tuple': (0.10, 0.02),
            'cpu-migrations': (35, 5),
            'dTLB-load-misses per Tuple': (2.4, 0.2)
            #'context-switches per Tuple': (0.0005, 0.0001)
        }
    }

    y_config = None
    if "indep" in programName:
        y_config = y_settings["indep"]
    elif "conc" in programName:
        y_config = y_settings["conc"]


    #loop through each metric and create individual plots
    for col, ylabel in metrics:
        #set plot fig size (same as tuples per second graphs)
        fig, ax = plt.subplots(figsize = (12,7))

        for thread in sorted(merged_df['threads'].unique()):
            color = thread_colors.get(thread, 'black')#defaults to black if no color is set in dictionary top of file.
            marker = thread_markers.get(thread, 'o')#defaults to 0 marker if no marker present in dict on top.
            subset = merged_df[merged_df['threads'] == thread] 
            #use plot with marker "o" and line style "-" to connect dots.
            ax.plot(subset['hashbits'], subset[col],
                    marker=marker, markersize=thread_markersize.get(thread, 8), linestyle='--', linewidth=2,
                    label=f'{thread} thread(s)', color=color)
            
        #set titles of plots/labels
        #plot_title = f'{ylabel} vs. Hashbits\n({programName} - {date})'
        #ax.set_title(plot_title, fontweight='bold')
        ax.set_xlabel('Hashbits')
        ax.set_ylabel(ylabel)
        ax.legend(title="Threads") # Add a title to the legend
        ax.grid(True, linestyle='--', alpha=0.7)
            
        max_hashbit = int(csvData['hashbits'].max())
        ax.set_xlim(1, max_hashbit)
        ax.set_xticks(range(1, max_hashbit + 1))

        min_y_value = merged_df[col].min()#find the minimum value for the metric

        if y_config and col in y_config:
            max_y, tick_interval = y_config[col]
            rounded_min_y = round_down(min_y_value, tick_interval)
            ax.set_ylim(bottom=min_y_value, top=max_y)
            ax.set_yticks(np.arange(rounded_min_y, max_y + tick_interval, tick_interval))
            
        plt.tight_layout()
        safe_col_name = col.lower().replace(' ', '_').replace('/', '_per_').replace('-', '_')

        output_filename = f'{date}_{programName}_{safe_col_name}.png'
        output_path = os.path.join(plots_results_folder, output_filename)

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_path}")

        # Close the current figure to free memory
        plt.close(fig)

def plot_tuples_persec_hashbits(merged_df, date, programName):
#compute throughput in millions of tuples/sec
    csvData = merged_df.copy()
    csvData['throughput_mtps'] = 16.777216 / csvData['wall_time']

    #only these threads, sorted by threads and hashbits
    sel_threads = [1, 2, 4, 8, 16, 32]
    csvData = (
        csvData[csvData['threads'].isin(sel_threads)]
        .sort_values(['threads', 'hashbits'])
    )

    #find maximum hashbit (so we can lock x-axis from 1 to 18)
    max_hashbit = int(csvData['hashbits'].max())

    #create figure + axis
    fig, ax = plt.subplots(figsize=(12, 7))

    #plot each result per threads
    for threads in sel_threads:
        subset = csvData[csvData['threads'] == threads]
        if subset.empty:
            continue
        ax.plot(
            subset['hashbits'],
            subset['throughput_mtps'],
            marker = thread_markers.get(threads, 'o'),#defaults to 0 marker if no marker present in dict on top.,
            linestyle='--',
            linewidth=2,
            markersize=thread_markersize.get(threads, 8),
            label=f'{threads} Threads',
            color=thread_colors.get(threads, 'black')
        )

    #labels & grid
    ax.set_xlabel('Hashbits', fontsize=12)
    ax.set_ylabel('Millions of tuples per second', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title='Thread Count', fontsize=10)

    #x-axis exactly 1 to max_hashbit, no side marginerinos
    ax.set_xticks(range(1, max_hashbit + 1))
    ax.set_xlim(1, max_hashbit)

    #set top‐of‐y based on programName, then apply both limits
    if "indep" in programName:
        y_top = 235
    elif "conc" in programName:
        y_top = 235

    ax.set_ylim(bottom=0, top=y_top)

    #custom yticks every 10 units
    ax.set_yticks(np.arange(0, y_top + 1, 10))

    plt.tight_layout()

    #saving graph to picture/png
    output_path = os.path.join(
        plots_results_folder,
        f'{date}_{programName}_performance_plot.png'
    )
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot: {output_path}")

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
    if "indep" in perf_filename:  #independent
        prefix = "indep"
    elif "conc" in perf_filename:  #concurrent
        prefix = "conc"
    else:
        prefix = "output"  #default if none of them are present.

    #check for core_aff_1 or core_aff_2 and append it to the prefix
    if "core_aff_1" in perf_filename:
        prefix += "_core_aff_1"
    elif "core_aff_2" in perf_filename:
        prefix += "_core_aff_2"

    #regular expression to find date at start of filename
    date_match = re.search(r'(\d{2}_\d{2})', perf_filename)
    if date_match:
        today_str = date_match.group(1)  #extract the date from file name
    else:
        today_str = datetime.today().strftime('%d_%m')  #if no date is found, use today's date.

    output_filename = f"output_{prefix}_{today_str}.csv"

    #the path to where the output csv is saved to.
    output_csv_path = os.path.join(csv_results_folder, output_filename)

    #write the dataframe to CSV
    merged_df.to_csv(output_csv_path, index=False)

    plot_hashbits_versus_perf_vals(merged_df, today_str, prefix)
    plot_tuples_persec_hashbits(merged_df, today_str,prefix)



#def plotting (results_perf, results_time):


    