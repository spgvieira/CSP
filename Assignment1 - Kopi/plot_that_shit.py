import pandas as pd
import matplotlib.pyplot as plt

#read the csv data
csvData = pd.read_csv("C:\computer systems performance\CSP\Assignment1\output_conc_1.csv")

#here we filter the stuff we want from csv
sel_threads = [1, 2, 4, 8, 16, 32]
csvData = csvData[csvData['threads'].isin(sel_threads)].sort_values(['threads', 'hashbits'])

#creating the plot
plt.figure(figsize=(12,7))

#plot a line for each thread count 1-32:
for threads in sel_threads:
    subset = csvData[csvData['threads'] == threads]
    if not subset.empty:
        plt.plot(subset['hashbits'],
                 subset['time_elapsed'],
                 marker='o',
                 linestyle='-',
                 linewidth=2,
                 markersize=8,
                 label=f' {threads} Threads')
        
#appearance of te plot graph
plt.xlabel('hashbits', fontsize=12)
plt.ylabel('Time elapsed (secs)', fontsize = 12)
plt.title('Time elapsed (perf) vs Hashbits by thread count.', fontsize = 14)
plt.xticks(range(1, 19)) #hashbits from 1 to 18
plt.grid(True, linestyle='--', alpha=0.7)

plt.legend(title='thread Count', fontsize=10)
plt.tight_layout()

#save plot
plt.savefig('performance_plot.png', dpi=300)
plt.show()