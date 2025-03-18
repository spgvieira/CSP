import pandas as pd
import matplotlib.pyplot as plt
import argparse

#argparse setup
parser = argparse.ArgumentParser(description='Plot performance/throughput from a CSV file.')
parser.add_argument('csv_file', type=str, help='Path to the input CSV file')
args = parser.parse_args()

#read CSV data (change to your csv filename or path)
csvData = pd.read_csv(args.csv_file)

#filter the dataframe to only include threads 16 and 32.
#csvData = csvData[csvData['threads'].isin([16, 32])]

# Convert columns for plotting clarity:
# Convert cpu-cycles to billions and cache-misses to millions.
csvData['cpu-cycles (B)'] = csvData['cpu-cycles'] / 1e9
csvData['cache-misses (M)'] = csvData['cache-misses'] / 1e6
csvData['dTLB-load-misses (M)'] = csvData['dTLB-load-misses'] / 1e6
csvData['page-faults (M)'] = csvData['page-faults'] / 1e6

#define mapping of thread counts to colors.
thread_colors = {
    1: 'cyan',
    2: 'green',
    4: 'purple',
    8: 'yellow',
    16: 'blue',
    32: 'red'
}

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
    for thread in sorted(csvData['threads'].unique()):
        color = thread_colors.get(thread, None)
        subset = csvData[csvData['threads'] == thread]
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
plt.savefig('hashbits_vs_perfvals.png', dpi=300)
plt.show()
