import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import re #for naming output files
from datetime import datetime #for naming output files
import matplotlib.ticker as ticker #for more stylish/even numbered tickers along y-axis

#define where the output plot images will be saved
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir) #go up one level if script is in a plots folder itself
plots_results_folder = os.path.join(project_dir, 'plots') #defines output folder name

#create plots folder if it doesn't exist
if not os.path.exists(plots_results_folder):
    os.makedirs(plots_results_folder)

#get todays date for filenames
today_str = datetime.today().strftime('%Y%m%d_%H%M%S')

#helper function to extract date prefix from filename
def extract_date_prefix(filename):
    #look for DD_MM or YYYYMMDD patterns
    match_dd_mm = re.search(r'^(\d{2}_\d{2})', filename)
    match_yyyymmdd = re.search(r'^(\d{8})', filename)
    if match_dd_mm:
        return match_dd_mm.group(1)
    elif match_yyyymmdd:
        # Optionally reformat YYYYMMDD to DD_MM if preferred
        # date_obj = datetime.strptime(match_yyyymmdd.group(1), '%Y%m%d')
        # return date_obj.strftime('%d_%m')
        return match_yyyymmdd.group(1) # Keep YYYYMMDD for now
    return None # Return None if no date pattern found

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

#this part down here processes any sequential data stuff
def calculate_averages_seq(csv_filepath):
    #reads a sequential CSV (input, time1, time2...)
    #and then calculates the average time for each input size row
    #load the csv as a dataframe
    df = pd.read_csv(csv_filepath, index_col='input')

    time_columns = df.columns #assumes all remaining columns are time data compared to input.
    averages = df[time_columns].mean(axis=1) #calculate mean across columns for each row (input size) using dataframe lib.
    return averages #returns those averages for each row

def plot_comparison_seq(averages1, averages2, label1, label2, base_filename, date_prefix):
    #plots the comparison for sequential data and saves it in one graph

    #just in case... checks to see both input sizes are present in both csv's using intersection to see
    common_index = averages1.index.intersection(averages2.index).sort_values()

    #filter the data from both csvs to only include the common inputs from both csv's.
    #e.g if the only input size/argument both csv's have is 5000, then the final graph will only show two columns comparing time
    #for input size/argument 5000!
    avg1_filtered = averages1.loc[common_index]
    avg2_filtered = averages2.loc[common_index]

    #basic plot setup stuff
    x = np.arange(len(common_index))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 7))

    #draws the bars - smart trick is to slightly offset them so they are side-by-side and easier to compare!
    rects1 = ax.bar(x - width/2, avg1_filtered, width, label=label1, color='skyblue')
    rects2 = ax.bar(x + width/2, avg2_filtered, width, label=label2, color='lightcoral')

    #labels, title, and ticks for the plot
    ax.set_ylabel('Average Wall Time (ms)') # Added (s) for clarity
    ax.set_xlabel('Input Size')
    ax.set_xticks(x)
    ax.set_xticklabels(common_index)
    ax.legend() #show the legend using the labels provided

    #make layout more neat and add a grid like in og csv_and_plot from ass1.
    fig.tight_layout()
    plt.xticks(rotation=45, ha='right') #rotate labels if they get crowded
    plt.grid(axis='y', linestyle='--', alpha=0.7) #add horizontal grid lines


    #automatically determine nice, rounded tick locations on the y-axis
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=20, prune='both', integer=True)) 

    #save the plot to a file
    clean_base = re.sub(r'_?sequential_?', '', base_filename, flags=re.IGNORECASE).strip('_-. ')
    if not clean_base: #fallback if cleaning removed everything
        clean_base = "sequential_data"

    filename_parts = []
    if date_prefix:#just incase file doesn't have a date prefix.
        filename_parts.append(date_prefix)
    filename_parts.append("imp_vs_func") #fixed string - might change later to also include the program/algo name.
    filename_parts.append("seq_comparison.png")

    output_filename = "_".join(filter(None, filename_parts)) #filter removes empty strings if date_prefix is none

    #remove potential double underscores if date_prefix was Nnne and clean_base started or ends with _
    output_filename = output_filename.replace("__", "_")

    output_path = os.path.join(plots_results_folder, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved sequential plot: {output_path}")
    plt.close(fig) #close the figure


#parallel part
def calculate_averages_parallel(csv_filepath):
    #reads a parallel CSV (input, thread, time1, time2...) and
    #calculates the average time for each (input, thread) combo
    df = pd.read_csv(csv_filepath)


    #this assumes that the "time" columns start from the 3rd position in the csv aka index 2.
    time_columns = df.columns[2:]

    #calculate the average time for each row first
    df['avg_time'] = df[time_columns].mean(axis=1)

    #twe group by both input AND thread - then find the mean
    #of the already calculated avg_time (this part is only really useful IF we say run input 1000 with the same amount of threads
    #MULTIPLE times - it finds the mean then of those average times. But currently it just returns the same as avg_time due to only
    #running same input and thread once.
    grouped_averages = df.groupby(['input', 'thread'])['avg_time'].mean()
    return grouped_averages


# NEW parallel plotting function using line graphs
def plot_performance_lines(grouped_averages, data_label, base_filename, date_prefix, max_y_value):
    """
    generates a line plot for a single parallel dataset, imperative or functional.
    x-axis: argument/input size
    y-axis: average time in ms
    lines: Different thread counts

    args:
        grouped_averages (pd.Series): multi-indexed Series (input, thread) -> avg_time.
        data_label (str): Label for the dataset (e.g., filename, "Imperative").
        base_filename (str): Base name for the output plot file.
        max_y_value (float): The maximum y-value for the plot axis (for normalization).
    """

    #get unique sorted thread counts and input sizes from the multi-index
    thread_counts = sorted(grouped_averages.index.get_level_values('thread').unique())
    input_sizes = sorted(grouped_averages.index.get_level_values('input').unique())

    min_input_size = min(input_sizes)
    max_input_size = max(input_sizes)

    #basic plot setup stuff
    fig, ax = plt.subplots(figsize=(12, 7))

    #plot a line for each thread count
    for thread in thread_counts:

        
        #extract data for the current thread (Series indexed by input_size)
        # Unstacking or selecting works. xs is concise here.
        # Need to handle cases where a thread might not have data for all input sizes
        if thread in grouped_averages.index.get_level_values('thread'):
            thread_data = grouped_averages.xs(thread, level='thread').reindex(input_sizes).sort_index()
        else:
            continue # Skip if this thread somehow isn't in the main index

        #get color and marker, use defaults if not defined
        color = thread_colors.get(thread, 'black') # Default to black
        marker = thread_markers.get(thread, '.')   # Default to point marker

        #plot the line for this thread
        ax.plot(thread_data.index, thread_data.values,
                marker=marker,
                linestyle='--', #dashed lines
                linewidth=2,
                markersize=8,  #standard marker size for lines
                label=f'{thread} Thread(s)',
                color=color)

    #labels, title, and ticks for the plot
    ax.set_ylabel('Average Wall Time (ms)')
    ax.set_xlabel('Arguments')
    #ax.set_title(f'Parallel Performance: {data_label}\nTime vs. Argument per Thread Count')

    #set x-axis ticks to be the actual input sizes
    ax.set_xticks(input_sizes)

    #set precise x-axis limits, e.g we don't start at 0.
    ax.set_xlim(left=min_input_size, right=max_input_size)

    ax.legend(title="Threads") #show the legend using the labels provided

    #set y-axis limit based on the calculated maximum across both datasets
    ax.set_ylim(bottom=0, top=max_y_value * 1.05) #add 5% padding to the top

    #it looks better if you add a tiny padding to y_max
    y_axis_max = max_y_value * 1.05

    #automatically determine nice rounded tick locations on the y-axis -
    #try aiming for around 10 ticks, prune removes ticks too close to edges
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=20, prune='both', integer=True)) #use integer=True if times are whole ms

    #make layout more neat and add a grid
    fig.tight_layout()
    ax.grid(True, linestyle='--', alpha=0.7) #add grid lines


    #clean up base_filename further if needed, remove date if extracted
    if date_prefix:
        base_filename_cleaned = base_filename.replace(date_prefix, "").strip('_-. ')
    else:
        base_filename_cleaned = base_filename
    #save the plot to a file
    #modify filename to indicate line plot and dataset
    filename_parts = []
    if date_prefix:
        filename_parts.append(date_prefix)
    filename_parts.append(data_label) #use provided clean label
    filename_parts.append("vs_argument_per_thread.png")

    output_filename = "_".join(filename_parts)
    #remove potential double underscores
    #output_filename = output_filename.replace("__", "_")

    output_path = os.path.join(plots_results_folder, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"saved parallel line plot: {output_path}")
    plt.close(fig) #close the figure


if __name__ == "__main__":
    #basic check for correct number of command-line arguments
    if len(sys.argv) != 3:
        script_name = os.path.basename(sys.argv[0])
        print(f"Usage: python {script_name} <csv_file_1> <csv_file_2>")
        print("Example: python plot_script.py imperative_results.csv functional_results.csv")
        print("\nFiles must both contain 'sequential' or 'parallel' in their names.")
        sys.exit(1)

    #grab the file paths from the command line
    csv_file1 = sys.argv[1]
    csv_file2 = sys.argv[2]

    # --- Filename Processing ---
    basename1 = os.path.basename(csv_file1)
    basename2 = os.path.basename(csv_file2)

    # Extract date prefix (preferring file1, fallback to file2, then today)
    date_prefix = extract_date_prefix(basename1)
    if not date_prefix:
        date_prefix = extract_date_prefix(basename2)
    # Use today's date only if needed later, maybe not for filename prefix itself

    # Use filename without extension for labels
    label1_base = os.path.splitext(basename1)[0]
    label2_base = os.path.splitext(basename2)[0]

    # Remove date prefix from labels if found
    label1_cleaned = label1_base.replace(date_prefix + "_", "") if date_prefix else label1_base
    label2_cleaned = label2_base.replace(date_prefix + "_", "") if date_prefix else label2_base

    # Determine file type based on cleaned labels (lowercase)
    file_type = None
    if "sequential" in label1_cleaned.lower() and "sequential" in label2_cleaned.lower():
        file_type = "sequential"
        # Further clean labels by removing "sequential"
        label1_final = label1_cleaned.lower().replace("_sequential", "").replace("sequential", "").strip('_-. ')
        label2_final = label2_cleaned.lower().replace("_sequential", "").replace("sequential", "").strip('_-. ')
    elif "parallel" in label1_cleaned.lower() and "parallel" in label2_cleaned.lower():
        file_type = "parallel"
         # Further clean labels by removing "parallel"
        label1_final = label1_cleaned.lower().replace("_parallel", "").replace("parallel", "").strip('_-. ')
        label2_final = label2_cleaned.lower().replace("_parallel", "").replace("parallel", "").strip('_-. ')
    else:
        #if mismatch or keywords missing, just stop
        print("\nError: Files must both contain 'sequential' or both contain 'parallel' in their names.")
        print(f"       File 1: {basename1}")
        print(f"       File 2: {basename2}")
        sys.exit(1)

    # Create a base name for sequential comparison (less critical now)
    base_plot_name_seq = os.path.commonprefix([label1_final, label2_final]).rstrip('_-. ')
    if not base_plot_name_seq:
        base_plot_name_seq = label1_final # Fallback

    # --- Processing and Plotting ---
    if file_type == "sequential":
        print(f"\nProcessing Sequential Files:\n - {basename1}\n - {basename2}")
        averages1 = calculate_averages_seq(csv_file1)
        averages2 = calculate_averages_seq(csv_file2)
        # Use the cleaned labels and base name for plotting
        plot_comparison_seq(averages1, averages2, label1_final, label2_final, base_plot_name_seq, date_prefix)

    elif file_type == "parallel":
        print(f"\nProcessing Parallel Files:\n - {basename1}\n - {basename2}")
        averages1 = calculate_averages_parallel(csv_file1) # Returns time in ms
        averages2 = calculate_averages_parallel(csv_file2) # Returns time in ms

        #calculate the overall maximum average time across BOTH datasets
        max_time1_ms = averages1.max() 
        max_time2_ms = averages2.max() 
        global_max_y_ms = max(max_time1_ms, max_time2_ms)

        print(f"   Global Max Avg Time (for Y-axis): {global_max_y_ms:.2f} ms")

        #plot each dataset separately using the new line plot function
        # Pass the final cleaned labels and the date prefix
        plot_performance_lines(averages1, label1_final, label1_final, date_prefix, global_max_y_ms)
        plot_performance_lines(averages2, label2_final, label2_final, date_prefix, global_max_y_ms)


    print("\nDone plotting. Can I go home now? I wanna play Minecraft!!!")