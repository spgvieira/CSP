# Assignment 1: Data Partitioning and Parallel Processing on Multicore CPUs

This project automates the benchmarking of multiple programs using Linux perf tool to gather performance metrics. It evaluates different threading and hash bit configurations under various scheduling and core affinity settings.

Project made by Su Mei Gwen Ho (suho@itu.dk), Sara Vieira (sapi@itu.dk) & Sophus Kaae Merved (some@itu.dk)

## Prerequisites
Ensure the following are installed on your system:
* perf (Linux Performance Counters tool)
* make
* All required executables are built and located in ./build/ (e.g., ./build/concurrent, ./build/independent, etc.)

## Usage
To run benchmarks, simply use:
`make <target>`
Available Targets
* all: Builds all executable test cases (assumes build system handles this).
* independent: Runs tests on independent program with various thread counts and hash bit sizes.
* indep_core_aff_1: Similar to independent but with core affinity strategy 1.
* indep_core_aff_2: Similar to above but with core affinity strategy 2 (uses THREADS_CORE_AFF_2 values).
* concurrent: Runs tests on concurrent program.
* conc_core_aff_1: Concurrent with core affinity strategy 1.
* conc_core_aff_2: Concurrent with core affinity strategy 2.

## Output

All results are saved under the results/ directory with filenames prefixed by the current day and month (e.g., 23_05_independent_results.txt).


## Plotting Graphs
## Usage of csv_and_plot.py

This script parses performance and wall time data files, merges them into a CSV, and generates comparison plots.

**1. Running the script:**

**Example:**
In the Assigment1 directory, run:
```bash
python ./data_analysis_tools/csv_and_plot.py ./results/indep_21_05.txt ./results/indep_21_05_time.txt
```

This processes the specified performance and time files, creates an output CSV, and generates plots comparing different metrics against Hashbits.
*   `<perf_file>`: Required. The path to the performance counter results text file (e.g., `indep_21_05.txt.txt`). The script extracts the date (e.g., `21_05`) and program type (e.g., `indep`, `conc`) from this filename to name output files.
*   `<time_file>`: Required. The path to the wall time results text file (e.g., `indep_21_05_time.txt`). This file should correspond to the same experiment as the performance file.

**2. Configuration Settings for csv_and_plot.py:**

You can modify the following dictionaries and variables directly within the `plot_script_v2.py` file (look for these definitions near the top) to customize plot appearance and which data is included:

*   `csv_results_folder`: Defines the directory where the combined CSV file is saved.
*   `plots_results_folder`: Defines the directory where the plot images (.png) are saved.
*   `thread_colors`: Dictionary mapping thread counts to the colors used for their lines on plots.
*   `thread_markers`: Dictionary mapping thread counts to the marker styles used for their data points.
*   `thread_markersize`: Dictionary mapping thread counts to the size of their data point markers.
*   `metrics` list (inside `plot_hashbits_versus_perf_vals` function): A list of tuples `(column_name, plot_label)` that determines which specific performance metrics get generated as individual plots.
*   `y_settings` dictionary (inside `plot_hashbits_versus_perf_vals` function): Controls the maximum Y-axis values and tick intervals for the performance metric plots, based on the program type (`indep` or `conc`) derived from the input filename.
*   `sel_threads` list (inside `plot_tuples_persec_hashbits` function): Defines which specific thread counts will have their data plotted on the "Millions of tuples per second" graph.
*   `y_top` (inside `plot_tuples_persec_hashbits` function): Sets the fixed maximum Y-axis value for the "Millions of tuples per second" plot, based on program type.

Modify these settings in the script file to tailor the output plots.