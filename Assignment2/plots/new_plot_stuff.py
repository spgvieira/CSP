import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import re #for naming output files
from datetime import datetime #for naming output files
import matplotlib.ticker as ticker #for more stylish/even numbered tickers along y-axis

# Code made by Su Mei Gwen Ho (suho@itu.dk), Sara Vieira (sapi@itu.dk) & Sophus Kaae Merved (some@itu.dk) with inspiration from Google's Gemini LLM

#=== Arguments ===
#argument handling
if len(sys.argv) < 2 or sys.argv[1] not in ['spec', 'mand']:
    print("Usage: python new_plot_stuff.py [spec|mand]")
    sys.exit(1)

#1st argument - map input to subfolder names
arg_map = {
    "spec": "spectral_norm",
    "mand": "mandelbrot"
}

#2nd and 3rd arguments for script, e.g select which parallel and sequential files to use based on date.
parallel_date = sys.argv[2]
sequential_date = sys.argv[3] if len(sys.argv) > 3 else None

#=== Folders and file paths ===

#=== Input folders ===
#selected subfolder based on 1st. argument
selected_subfolder = arg_map[sys.argv[1]]
#define folder paths relative to the script location
#base folder setup
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(base_dir, "..", "app"))

PLOTS_FOLDER_MAP = {
    "time": "plots_time",
    "perf": "plots_perf",
    "mem": "plots_mem"
}

#define paths to each data category selected subfolder (spectral norm or mandelbrot)
time_folder = os.path.join(parent_dir, "experimental_results", selected_subfolder)
perf_folder = os.path.join(parent_dir, "perf_reports", selected_subfolder)
mem_folder = os.path.join(parent_dir, "mem_reports", selected_subfolder)
#pidmem_folder = os.path.join(parent_dir, "mem_pid_reports", selected_subfolder)

#=== Values for graph customization ===
# Define thread colors
thread_colors = {
    1: 'cyan',
    2: 'green',
    4: 'magenta',
    8: 'orange',
    16: 'blue',
    32: 'red',
    64: 'pink',
    128: 'lightblue'
}

# Define thread markers
thread_markers = {
    1:  'o',
    2:  's',
    4:  '^',
    8:  'D',
    16: 'X',
    32: '*'
}

# Define marker size
thread_markersize = {
    1: 8,   #make the circle marker noticeably bigger
    2: 8,
    4: 8,
    8: 8,
    16: 8,
    32: 8
}

#dictionary to control what to plot

metrics_to_divide_with_input_size = {
    "cache_misses",
    #"dtlb_load_misses",
    #"page_faults",
    #"cpu_cycles"
}

#dictionary to control which perf metrics to plot
perf_metrics_collection = {
    "task_clock_msec": False,
    "context_switches": True,
    "cpu_migrations": False,
    "page_faults": True,
    "cycles": False,
    "cpu_cycles": True,
    "instructions": True,
    "branches": False,
    "branch_misses": False,
    "time_elapsed_sec": False,
    "user_time_sec": False,
    "sys_time_sec": False,
    "major_faults": False,
    "dtlb_load_misses": True,
    "cache_misses": True,
    "ipc": True
}

#dictionary to control which memory metrics to plot
mem_metrics_collection = {
    "free_mem": True,
    'pid_mem': True
}

#define which metrics you don't want to scale y-graph between imp. and func:
indep = {
    "context_switches"
}

#Other values:
#baseline_mem = 0 #calculated by taking sum of all baseline values and dividing by 60.
baseline_subtract_true = True #subtract or not from free/total memory.
baseline_mem_mandelbrot = round(49951697.370787)
baseline_mem_spec_18_05 = round(50058742.58427)
baseline_mem_spec_19_05 = round(46986934.368715)

#enable log scale for wall_time?:
log_scale = False
#change log base to something else, e.g 2, 4, 6, etc.
log_base = 10

#=== HELPER FUNCTIONS ===
#filters files based on date and keyword (e.g parallel or sequential)
def filter_files(files, date_str, keyword):
    return [f for f in files if os.path.basename(f).startswith(date_str) and keyword in os.path.basename(f).lower()]

#helper function to split whether they are functional or imperative
def split_by_style(files, styles=('functional', 'imperative')):
    result = {style: [] for style in styles}
    for f in files:
        basename = os.path.basename(f).lower()
        for style in styles:
            if style in basename:
                result[style].append(f)
    return result

#helper function for listing all files inside a folder, and returns them as a list. 
def list_files(folder_path, extensions=None):
    if not os.path.isdir(folder_path):
        print(f"Warning: Folder {folder_path} not found.")
        return []
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
             if os.path.isfile(os.path.join(folder_path, f))]
    if extensions:
        files = [f for f in files if f.lower().endswith(tuple(ext.strip() for ext in extensions))]
    return files

#calculates the average time per inputsize from a sequential csv time file
def calculate_averages_seq(time_csv_filepath):
    #reads a sequential CSV (input, time1, time2...)
    #and then calculates the average time for each input size row
    #load the csv as a dataframe
    df = pd.read_csv(time_csv_filepath, index_col='input')

    time_columns = df.columns #assumes all remaining columns are time data compared to input.
    averages = df[time_columns].mean(axis=1) #calculate mean across columns for each row (input size) using dataframe lib.
    return averages #returns those averages for each row

#calculates the average time per inputsize/thread combo from a parallel csv time file
def calculate_averages_parallel(time_csv_filepath):
    #reads a parallel CSV (input, thread, time1, time2...) and
    #calculates the average time for each (input, thread) combo
    df = pd.read_csv(time_csv_filepath)


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


def compute_max_y(parallel_csvs, sequential_csvs=None, metrics=None):
    """
    Determine maximum grouped mean across parallel and sequential CSV files.
    parallel_csvs: list of file paths
    sequential_csvs: list of file paths (optional)
    metrics: list of column names to consider; e.g. ['avg_time'] or ['cycles','instructions']
    Returns: dict of {metric: max_value} if multiple metrics, or single float if len(metrics)==1
    """
    if metrics is None:
        metrics = ['avg_time']
    max_vals = {m: 0 for m in metrics}

    # Process parallel CSVs
    for csv in parallel_csvs:
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        for m in metrics:
            if m == 'avg_time':
                grp = calculate_averages_parallel(csv)
                max_vals[m] = max(max_vals[m], grp.max())
            elif m in df.columns:
                grp = df.groupby(['input', 'thread'])[m].mean()
                max_vals[m] = max(max_vals[m], grp.max())

    # Process sequential CSVs
    if sequential_csvs:
        for csv in sequential_csvs:
            try:
                df = pd.read_csv(csv)
            except Exception:
                continue
            for m in metrics:
                if m == 'avg_time':
                    seq = calculate_averages_seq(csv)
                    max_vals[m] = max(max_vals[m], seq.max())
                elif m in df.columns:
                    if 'thread' in df.columns:
                        grp = df.groupby(['input', 'thread'])[m].mean()
                    else:
                        grp = df.groupby('input')[m].mean()
                    max_vals[m] = max(max_vals[m], grp.max())

    return max_vals if len(metrics) > 1 else max_vals[metrics[0]]

def compute_max_or_min_y_norm(parallel_csvs, sequential_csvs=None, metrics=None, metrics_to_normalize=None, aggregate_type='max'):
    """
    Determine maximum or minimum grouped mean across parallel and sequential CSV files.
    Can optionally normalize metrics by input size squared BEFORE finding mean.

    parallel_csvs: list of file paths
    sequential_csvs: list of file paths (optional)
    metrics: list of column names to consider
    metrics_to_normalize: set or list of metric names to normalize by input**2
                          BEFORE finding the mean.
    aggregate_type: 'max' or 'min' - determines whether to find the maximum or minimum
                    of the grouped means/normalized means.
    Returns: dict of {metric: value} or single float if len(metrics)==1
             Returns {metric: None} for metrics not found or having no data after aggregation.
    """
    if metrics is None:
        metrics = ['avg_time']
    if metrics_to_normalize is None:
        metrics_to_normalize = set() # Use a set for efficient lookup
    else:
         metrics_to_normalize = set(metrics_to_normalize) # Convert to set

    # Input validation for aggregate_type
    if aggregate_type not in ['max', 'min']:
        print(f"Error: Invalid aggregate_type '{aggregate_type}'. Must be 'max' or 'min'.")
        return None # Or raise an error

    # Initialize vals based on aggregate_type
    initial_val = float('-inf') if aggregate_type == 'max' else float('inf')
    vals = {m: initial_val for m in metrics}
    found_any_data = {m: False for m in metrics} # Track if we found *any* valid aggregated value for a metric

    # Helper to update a metric’s value
    def _update_value(m, new_value):
        # Ensure new_value is a finite number before comparing
        if isinstance(new_value, (int, float)) and not np.isnan(new_value) and not np.isinf(new_value):
             if aggregate_type == 'max':
                  # Initialize vals[m] if it's still the initial extreme value
                  if vals[m] == float('-inf') or new_value > vals[m]:
                     vals[m] = new_value
                     found_any_data[m] = True
             else: # aggregate_type == 'min'
                  # Initialize vals[m] if it's still the initial extreme value
                  if vals[m] == float('inf') or new_value < vals[m]:
                     vals[m] = new_value
                     found_any_data[m] = True


    # Function to process a single DataFrame (either parallel or sequential)
    def process_df(df, is_parallel):
        if df.empty:
            return # Nothing to process

        # Ensure 'input' column exists if any metric needs normalization
        needs_input_for_norm = any(m in metrics_to_normalize for m in metrics)
        if needs_input_for_norm and 'input' not in df.columns:
             print(f"Warning: 'input' column missing in file processed by compute_max_or_min_y_norm. Cannot normalize metrics in this file.")
             # Don't clear the global metrics_to_normalize, just skip for this df

        # Calculate grouped means for each metric
        for m in metrics:
            if m not in df.columns or not pd.api.types.is_numeric_dtype(df[m]):
                continue # Metric not in this file or not numeric

            # Determine grouping keys
            group_keys = ['input']
            if is_parallel and 'thread' in df.columns:
                group_keys.append('thread')
            elif is_parallel and 'thread' not in df.columns: # Parallel file without thread column?
                 print(f"Warning: Parallel file without 'thread' column. Grouping by input only for metric '{m}'.")

            # Calculate grouped metric value (mean of potentially normalized value)
            if m in metrics_to_normalize and 'input' in df.columns and (df['input'] != 0).any():
                # --- Handle normalization before grouping ---
                # Create a temporary normalized series, handle input = 0
                normalized_series = pd.Series(np.nan, index=df.index)
                non_zero_input_mask = df['input'] != 0
                # Ensure metric column is numeric before division
                if pd.api.types.is_numeric_dtype(df[m]):
                    normalized_series.loc[non_zero_input_mask] = df.loc[non_zero_input_mask, m] / (df.loc[non_zero_input_mask, 'input'] ** 2)
                else:
                    print(f"Warning: Metric '{m}' is not numeric, cannot normalize.")
                    continue # Skip this metric for this file if not numeric


                # Group the temporary normalized series and find the mean
                # The series will be indexed by the group keys. Use mean() after grouping.
                if group_keys:
                    temp_df = df[group_keys].copy()
                    temp_df['normalized_metric'] = normalized_series
                    # Ensure the normalized_metric column is numeric after assignment
                    temp_df['normalized_metric'] = pd.to_numeric(temp_df['normalized_metric'], errors='coerce')
                    if not temp_df['normalized_metric'].dropna().empty: # Only group if there's valid data after norm/coerce
                        grp = temp_df.groupby(group_keys)['normalized_metric'].mean()
                    else:
                        grp = pd.Series([], dtype='float64') # Empty Series if no valid data after norm/coerce
                else: # Should not happen if input exists
                     # Ensure normalized_series is numeric
                     normalized_series = pd.to_numeric(normalized_series, errors='coerce')
                     if not normalized_series.dropna().empty:
                         grp = normalized_series.mean() # Just take the mean of the column
                     else:
                         grp = np.nan # Scalar NaN if no valid data


            else: # Not normalizing, or cannot normalize in this file
                # --- Handle metrics that are NOT normalized OR cannot be normalized in this file ---
                if group_keys:
                    # Ensure metric column is numeric before mean aggregation
                    if pd.api.types.is_numeric_dtype(df[m]):
                         grp = df.groupby(group_keys)[m].mean()
                    else:
                         print(f"Warning: Metric '{m}' is not numeric, cannot compute mean.")
                         continue # Skip this metric for this file if not numeric
                else: # e.g. no 'input' column for sequential file?
                     # Ensure metric column is numeric before mean aggregation
                     if pd.api.types.is_numeric_dtype(df[m]):
                         grp = df[m].mean() # Just take the mean of the column
                     else:
                         print(f"Warning: Metric '{m}' is not numeric, cannot compute mean.")
                         continue # Skip this metric for this file if not numeric


            # Find the overall max or min among the group means
            overall_agg_value = None # Initialize aggregate value for this metric/file

            if isinstance(grp, pd.Series):
                 if not grp.empty:
                      # Ensure we only consider finite numeric values when finding max/min
                      numeric_values = grp.dropna().replace([np.inf, -np.inf], np.nan).dropna()
                      if not numeric_values.empty:
                           overall_agg_value = numeric_values.max() if aggregate_type == 'max' else numeric_values.min()
            elif isinstance(grp, (int, float)): # Handle scalar result
                 # Check if scalar is valid before assigning
                 if not np.isnan(grp) and not np.isinf(grp):
                      overall_agg_value = grp

            # Update the global min/max for the metric if overall_agg_value is valid
            _update_value(m, overall_agg_value)


    # Process parallel CSVs
    for csv in parallel_csvs:
        try:
            df = pd.read_csv(csv)
            process_df(df, is_parallel=True)
        except FileNotFoundError:
             print(f"Warning: File not found: {csv}")
             continue
        except Exception as e:
            print(f"Error reading or processing {csv} in compute_max_or_min_y_norm: {e}")
            continue


    # Process sequential CSVs
    if sequential_csvs:
        for csv in sequential_csvs:
            try:
                df = pd.read_csv(csv)
                # Sequential files typically don't have a 'thread' column,
                # so pass is_parallel=False
                process_df(df, is_parallel=False)
            except FileNotFoundError:
                 print(f"Warning: File not found: {csv}")
                 continue
            except Exception as e:
                 print(f"Error reading or processing {csv} in compute_max_or_min_y_norm: {e}")
                 continue


    # Replace initial_val with None for metrics where no valid data was found
    # If the initial value is still the extreme one, it means no valid data point updated it.
    final_vals = {}
    for m in metrics:
        if found_any_data[m]:
            final_vals[m] = vals[m]
        else:
            final_vals[m] = None # Set to None if no valid data was found for this metric


    return final_vals if len(metrics) > 1 else final_vals.get(metrics[0], None)


def compute_peak_max_y(parallel_csvs, sequential_csvs=None, metrics=None):
    """
    Determine maximum grouped MAX across parallel and sequential CSV files for specific metrics.
    This is intended for metrics like PID memory where the peak is desired per group.
    parallel_csvs: list of file paths
    sequential_csvs: list of file paths (optional)
    metrics: list of column names to consider (e.g., ['pid_mem'])
    Returns: dict of {metric: max_peak_value} if multiple metrics, or single float if len(metrics)==1
             Returns None or {metric: None} if no data found.
    """
    if metrics is None:
        metrics = [] # Default to empty if no metrics specified
    max_vals = {m: float('-inf') for m in metrics}
    found_any_data = {m: False for m in metrics}

    all_csvs = parallel_csvs + (sequential_csvs if sequential_csvs else [])

    for csv in all_csvs:
        try:
            df = pd.read_csv(csv)
            df.columns = df.columns.str.strip() # Clean column names
        except Exception:
            continue # Skip unreadable files

        # Apply KB to MB conversion for memory metrics *before* aggregation
        # Assuming 'pid_mem' is in KB and needs conversion for comparison
        if 'pid_mem' in df.columns and pd.api.types.is_numeric_dtype(df['pid_mem']):
             df['pid_mem'] = df['pid_mem'] / 1024.0

        for m in metrics:
            if m in df.columns and pd.api.types.is_numeric_dtype(df[m]):
                current_max = float('-inf')
                # Group and find the MAX of the metric for each group
                if 'input' in df.columns:
                    if 'thread' in df.columns:
                        # Parallel structure - group by input and thread, take max of metric
                        grp = df.groupby(['input', 'thread'])[m].max()
                    else:
                        # Sequential structure - group by input, take max of metric
                        grp = df.groupby('input')[m].max()
                else:
                    # Cannot group by input, just take max of the whole column if numeric
                    current_max = df[m].max() # Use max of the column directly

                if 'grp' in locals() and not grp.empty:
                    # Find the overall maximum among the group maximums
                    numeric_values = grp.dropna().replace([np.inf, -np.inf], np.nan).dropna()
                    if not numeric_values.empty:
                         current_max = numeric_values.max()


                if not np.isnan(current_max) and not np.isinf(current_max):
                     max_vals[m] = max(max_vals[m], current_max)
                     found_any_data[m] = True

    # Replace -inf with None for metrics where no data was found
    final_max_vals = {m: (v if found_any_data[m] else None) for m, v in max_vals.items()}

    return final_max_vals if len(metrics) > 1 else final_max_vals.get(metrics[0], None)

def compute_peak_min_y(parallel_csvs, sequential_csvs=None, metrics=None):
    """
    Determine minimum grouped MAX across parallel and sequential CSV files for specific metrics.
    This is intended for metrics like PID memory where the peak is desired per group.
    parallel_csvs: list of file paths
    sequential_csvs: list of file paths (optional)
    metrics: list of column names to consider (e.g., ['pid_mem'])
    Returns: dict of {metric: min_peak_value} if multiple metrics, or single float if len(metrics)==1
             Returns None or {metric: None} if no data found.
    """
    if metrics is None:
        metrics = [] # Default to empty if no metrics specified
    min_vals = {m: float('inf') for m in metrics}
    found_any_data = {m: False for m in metrics}

    all_csvs = parallel_csvs + (sequential_csvs if sequential_csvs else [])

    for csv in all_csvs:
        try:
            df = pd.read_csv(csv)
            df.columns = df.columns.str.strip() # Clean column names
        except Exception:
            continue # Skip unreadable files

        # Apply KB to MB conversion for memory metrics *before* aggregation
        if 'pid_mem' in df.columns and pd.api.types.is_numeric_dtype(df['pid_mem']):
             df['pid_mem'] = df['pid_mem'] / 1024.0


        for m in metrics:
            if m in df.columns and pd.api.types.is_numeric_dtype(df[m]):
                current_min = float('inf')
                # Group and find the MAX of the metric for each group
                if 'input' in df.columns:
                    if 'thread' in df.columns:
                        # Parallel structure - group by input and thread, take max of metric
                        grp = df.groupby(['input', 'thread'])[m].max()
                    else:
                        # Sequential structure - group by input, take max of metric
                        grp = df.groupby('input')[m].max()
                else:
                    # Cannot group by input, just take min of the whole column if numeric
                    current_min = df[m].min() # Use min of the column directly

                if 'grp' in locals() and not grp.empty:
                     # Find the overall minimum among the group maximums
                     numeric_values = grp.dropna().replace([np.inf, -np.inf], np.nan).dropna()
                     if not numeric_values.empty:
                         current_min = numeric_values.min()


                if not np.isnan(current_min) and not np.isinf(current_min):
                     min_vals[m] = min(min_vals[m], current_min)
                     found_any_data[m] = True

    # Replace +inf with None for metrics where no data was found
    final_min_vals = {m: (v if found_any_data[m] else None) for m, v in min_vals.items()}

    return final_min_vals if len(metrics) > 1 else final_min_vals.get(metrics[0], None)

#minimum instead of max
def compute_min_y(parallel_csvs, sequential_csvs=None, metrics=None):
    """
    Determine minimum grouped mean across parallel and sequential CSV files.
    parallel_csvs: list of file paths
    sequential_csvs: list of file paths (optional)
    metrics: list of column names to consider; e.g. ['avg_time'] or ['cycles','instructions']
    Returns: dict of {metric: min_value} if multiple metrics, or single float if len(metrics)==1
    """
    if metrics is None:
        metrics = ['avg_time']
    # Initialize mins to +inf so any real value will be smaller
    min_vals = {m: float('inf') for m in metrics}

    # helper to update a metric’s min
    def _update_min(m, value):
        if value is not None and not np.isnan(value):
            min_vals[m] = min(min_vals[m], value)

    # Process parallel CSVs
    for csv in parallel_csvs:
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue

        for m in metrics:
            if m == 'avg_time':
                grp = calculate_averages_parallel(csv)
                _update_min(m, grp.min())
            elif m in df.columns:
                grp = df.groupby(['input', 'thread'])[m].mean()
                _update_min(m, grp.min())

    # Process sequential CSVs
    if sequential_csvs:
        for csv in sequential_csvs:
            try:
                df = pd.read_csv(csv)
            except Exception:
                continue

            for m in metrics:
                if m == 'avg_time':
                    seq = calculate_averages_seq(csv)
                    _update_min(m, seq.min())
                elif m in df.columns:
                    if 'thread' in df.columns:
                        grp = df.groupby(['input', 'thread'])[m].mean()
                    else:
                        grp = df.groupby('input')[m].mean()
                    _update_min(m, grp.min())

    # If a metric was never found, drop it (or you could leave inf)
    for m in metrics:
        if min_vals[m] == float('inf'):
            min_vals[m] = None

    return min_vals if len(metrics) > 1 else min_vals[metrics[0]]

#converts .txt files from perf to csv:
def parse_perf_data(input_path: str, output_csv: str):
    """
    Read a perf stat .data/.txt file and dump it as a clean CSV.
    Uses two regexes to capture:
      • Parallel runs: InputSize + Threads
      • Sequential runs: InputSize only (no Threads column)
    """
    # Regex for parallel runs (two numbers at end)
    par_re = re.compile(
        r"Performance counter stats for '.*\s+"
        r"(?P<InputSize>\d+)\s+"       # e.g. 1000
        r"(?P<Threads>\d+)"            # e.g. 1, 2, 4, ...
        r"'\s*:"
    )
    # Regex for sequential runs (one number at end)
    seq_re = re.compile(
        r"Performance counter stats for '.*\s+"
        r"(?P<InputSize>\d+)"          # e.g. 1000
        r"'\s*:"
    )

    #updated metric patterns to match new perf outputs from 07-04 and 08-04
    metric_pats = {
        "task_clock_msec":    re.compile(r"^\s*([\d,\.]+)\s+msec\s+task-clock"),
        "context_switches":   re.compile(r"^\s*([\d,]+)\s+context-switches"),
        "cpu_migrations":     re.compile(r"^\s*([\d,]+)\s+cpu-migrations"),
        "page_faults":        re.compile(r"^\s*([\d,]+)\s+page-faults"),
        "major_faults":       re.compile(r"^\s*([\d,]+)\s+major-faults"),
        "dtlb_load_misses":   re.compile(r"^\s*([\d,]+)\s+dTLB-load-misses"),
        "cache_misses":       re.compile(r"^\s*([\d,]+)\s+cache-misses"),
        "cpu_cycles":         re.compile(r"^\s*([\d,]+)\s+cpu-cycles"),
        "instructions":       re.compile(r"^\s*([\d,]+)\s+instructions"),
        "branches":           re.compile(r"^\s*([\d,]+)\s+branches"),
        "branch_misses":      re.compile(r"^\s*([\d,]+)\s+branch-misses"),
        "time_elapsed_sec":   re.compile(r"^\s*([\d\.]+)\s+seconds time elapsed"),
        "user_time_sec":      re.compile(r"^\s*([\d\.]+)\s+seconds user"),
        "sys_time_sec":       re.compile(r"^\s*([\d\.]+)\s+seconds sys"),
    }

    records = []
    current = None
    seen_parallel = False

    with open(input_path, 'r') as f:
        for raw in f:
            line = raw.strip()

            # Try parallel first
            m = par_re.search(line)
            if m:
                if current:
                    records.append(current)
                current = { k: None for k in ("InputSize","Threads", *metric_pats) }
                current["InputSize"] = int(m.group("InputSize"))
                current["Threads"]   = int(m.group("Threads"))
                seen_parallel = True
                continue

            # If not parallel, try sequential
            m2 = seq_re.search(line)
            if m2:
                if current:
                    records.append(current)
                current = { k: None for k in ("InputSize", *metric_pats) }
                current["InputSize"] = int(m2.group("InputSize"))
                seen_parallel = seen_parallel or False
                continue

            # Inside a block, pick up metrics
            if current:
                for name, pat in metric_pats.items():
                    m3 = pat.match(line)
                    if m3:
                        s = m3.group(1).replace(',', '')
                        current[name] = float(s) if '.' in s else int(s)
                        break

    # Append final block
    if current:
        records.append(current)

    if not records:
        print(f"[parse_perf_data] no records parsed from {input_path}")
        return

    df = pd.DataFrame(records)

    df['ipc'] = df['instructions'] / df['cpu_cycles']
    df['ipc'].replace([float('inf'), -float('inf')], 0, inplace=True)
    df['ipc'].fillna(0, inplace=True)

    # Rename the columns to match other functions:
    #   InputSize → input
    #   Threads   → thread
    rename_map = {"InputSize": "input", "Threads": "thread"}
    df.rename(columns=rename_map, inplace=True)

    # Build column list: input, thread (only if we saw any parallel), then metrics
    cols = ["input"]
    if seen_parallel:
        cols.append("thread")
    cols.extend(metric_pats.keys())
    cols.append('ipc')

    df.to_csv(output_csv, columns=cols, index=False)
    print(f"[parse_perf_data] wrote {len(df)} rows to {output_csv}")

def add_missing_column_names_and_clean(list_of_csv_paths):
    """
    For each CSV in `paths` (single path or list), ensure it has the correct header row
    (based on 'parallel' vs 'sequential' in filename) and clean memory columns from textual
    entries (e.g., 'MiB Mem : 15690.9 total') down to pure floats.
    """
    if isinstance(list_of_csv_paths, (list, tuple)):
        for p in list_of_csv_paths:
            add_missing_column_names_and_clean(p) # Recursive call
        return
    
    path = list_of_csv_paths # Now path is a single file path

    basename = os.path.basename(path).lower()
    is_parallel = 'parallel' in basename
    is_sequential = 'sequential' in basename

    # Determine expected headers
    if is_parallel:
        expected_headers = ['input', 'thread', 'free_mem', 'pid_mem']
    elif is_sequential:
        expected_headers = ['input', 'free_mem', 'pid_mem']
    else:
        print(f"Skipping {path}: cannot determine type (parallel/sequential) from filename.")
        return

    # Read file lines to check/add header
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {path}")
        return
    
    header_added_or_file_empty = False
    if not lines:
        print(f"Empty file: {path}. Adding headers.")
        header_line = ','.join(expected_headers) + '\n'
        lines.append(header_line) # Add header to empty file
        header_added_or_file_empty = True
        # Fall through to write this single-line (header-only) file
    else:
        # Check current header
        first_tokens = lines[0].strip().lower().split(',')
        if not (first_tokens and first_tokens[0] == 'input'):
            header_line = ','.join(expected_headers) + '\n'
            lines.insert(0, header_line)
            print(f"Added header to {path}")
            header_added_or_file_empty = True
        # else:
            # print(f"Header seems to exist in {path} (starts with 'input').")

    # If header was added, need to write lines back before pandas reads,
    # OR use io.StringIO if we want to avoid intermediate write.
    # The original code wrote, then re-read. Let's stick to that pattern for minimal change.
    if header_added_or_file_empty: # Only write if header was added or file was empty
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
        except Exception as e:
            print(f"Error writing header to {path}: {e}")
            return


    # Load with pandas to clean mem columns
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read CSV {path} into DataFrame: {e}")
        return
    
    if df.empty and not header_added_or_file_empty : # If df is empty and it wasn't an empty file we just put headers in
        print(f"DataFrame loaded from {path} is empty. Skipping cleaning. File might only contain headers or be malformed.")
        # If it was an empty file we added headers to, df will be empty, and that's fine.
        # We don't need to write it again unless other cleaning happens.
        if header_added_or_file_empty: # if it's an empty file we just wrote headers to
             # df.to_csv(path, index=False) # Ensures it's a valid CSV structure if it was just headers
             # print(f"Initialized {path} with headers.")
             return # Nothing to clean
        return


    # Regex to extract first float
    num_pat = re.compile(r"([0-9]+(?:\.[0-9]+)?)")
    def extract_num(s):
        if pd.isna(s):
            return s
        # Convert to string in case it's already a number (float/int)
        m = num_pat.search(str(s))
        return float(m.group(1)) if m else pd.NA # Use pd.NA for consistency

    # Determine which columns to clean based on file type (parallel/sequential)
    if is_parallel:
        # For parallel: ['input', 'thread', 'total_mem', 'free_mem', 'used_mem', 'buff_cache']
        # Memory columns start at index 2 of expected_headers
        cols_to_clean_names = expected_headers[2:]
    else: # is_sequential
        # For sequential: ['input', 'total_mem', 'free_mem', 'used_mem', 'buff_cache']
        # Memory columns start at index 1 of expected_headers
        cols_to_clean_names = expected_headers[1:]

    cleaned_at_least_one_column = False
    for col_name in cols_to_clean_names:
        if col_name in df.columns:
            # Check if column needs cleaning (i.e., is not already numeric)
            # This avoids errors if a column is already float/int
            if df[col_name].dtype == 'object': # 'object' dtype usually means strings
                df[col_name] = df[col_name].apply(extract_num)
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce') # Ensure numeric type
                cleaned_at_least_one_column = True
        # else:
            # This case should ideally not happen if headers were correctly added/matched
            # print(f"Warning: Expected column '{col_name}' not found in DataFrame from {path} for cleaning.")

    # Write back cleaned CSV
    if cleaned_at_least_one_column or header_added_or_file_empty : # Write if we cleaned something or if we added/fixed header initially
        try:
            df.to_csv(path, index=False)
            if cleaned_at_least_one_column :
                print(f"Cleaned memory columns in {path}")
            # If only header was added and no cleaning, the earlier message "Added header to {path}" suffices.
        except Exception as e:
            print(f"Failed to write cleaned CSV {path}: {e}")
    # else:
        # print(f"No cleaning performed or header modifications needed for {path}.")
        
    # The redundant block that was here is now removed.


#cleans out strings in columns of memory csv's
def clean_mem_col(series):
    # extract the first floating‑point number found in each cell
    return series.astype(str).str.extract(r"([0-9]+(?:\.[0-9]+)?)")[0].astype(float)

def round_to_nice_step(value, n_intervals=10):
    """
    Calculates a 'nice' step size for axis ticks based on a data value
    and desired number of intervals.
    """
    if value <= 0:
        return 1.0 # Default small step for zero or negative values

    # Estimate a raw step based on the value and desired intervals
    rough_step = value / n_intervals

    # Find the magnitude (power of 10) of the rough step
    exponent = np.floor(np.log10(rough_step))
    magnitude = 10**exponent

    # Normalize the rough step to be between 1 and 10
    fraction = rough_step / magnitude

    # Round the fraction to a 'nice' number (1, 2, 5)
    if fraction <= 1.5:
        rounded_fraction = 1
    elif fraction <= 3.5:
        rounded_fraction = 2
    elif fraction <= 7.5:
        rounded_fraction = 5
    else:
        # If it's > 7.5, use 10 (and increment the exponent)
        rounded_fraction = 10

    # Calculate the nice step size
    nice_step = rounded_fraction * magnitude

    return nice_step

def find_nice_axis_limits(data_min, data_max, nbins=10, min_floor=0.0, padding_percent_below=15.0, padding_percent_above=5.0):
    """
    Calculates nice, rounded axis limits (y0, y1) that enclose the data range,
    respect a minimum floor (like 0), are multiples of a 'nice' step size,
    and include specified padding percentages above and below the data range.
    Also returns the calculated nice step size.
    """
    # Ensure data_min and data_max are treated as numbers, handle None/inf/nan inputs gracefully
    # Convert potential input types (like None) to a value np.isnan/np.isinf can handle or default
    data_min_float = float(data_min) if isinstance(data_min, (int, float)) else np.nan
    data_max_float = float(data_max) if isinstance(data_max, (int, float)) else np.nan

    # Check for invalid inputs after conversion
    if np.isnan(data_min_float) or np.isnan(data_max_float) or np.isinf(data_min_float) or np.isinf(data_max_float):
        print(f"Warning: Invalid data range [{data_min}, {data_max}] for nice limits. Using default (min_floor, min_floor + 100).")
        # Calculate a default step based on a reasonable range
        default_range = 100.0
        step = round_to_nice_step(default_range, n_intervals=nbins) or 1.0
        return min_floor, min_floor + default_range, step

    # Ensure data_min is not greater than data_max (after handling NaNs/Infs)
    if data_min_float > data_max_float:
         print(f"Warning: data_min ({data_min_float}) > data_max ({data_max_float}). Swapping or using default range.")
         # If max is less than min, default to a sensible small range starting at min_floor
         # Calculate a minimal range based on the difference or a default if difference is non-positive
         min_range_needed = (data_min_float - data_max_float) * 1.1 + 1.0
         if min_range_needed <= 0: min_range_needed = 1.0 # Ensure positive range
         step = round_to_nice_step(min_range_needed / nbins, n_intervals=nbins) or 1.0 # Step based on minimal range
         if step <= 0: step = 1.0
         y0_final = min_floor
         y1_final = y0_final + step * nbins # Define upper limit based on step and nbins
         # Ensure y1 is strictly greater than y0
         if y1_final <= y0_final: y1_final = y0_final + step
         return y0_final, y1_final, step


    # Calculate the *full* data range that the axis should cover
    full_data_range = data_max_float - data_min_float

    # If the data range is zero or very small, handle as a special case
    if full_data_range <= 1e-9: # Use a small tolerance for floating point comparisons
         # If range is 0 or near zero, step should be based on the value itself or a default
         # Choose a step based on the magnitude of the value, if non-zero, or a default
         step_base_value = max(abs(data_max_float), 1.0) # Base for step calculation, use 1.0 if data is 0 or negative
         step = round_to_nice_step(step_base_value / nbins, n_intervals=nbins) or 1.0
         if step <= 0: step = 1.0 # Ensure step is positive

         # For zero range, center the data point within a range of roughly 2 steps.
         # Ensure y0 respects min_floor.
         target_y0_tiny = data_min_float - step # Aim one step below the data point
         y0_final_tiny = max(min_floor, np.floor(target_y0_tiny / step) * step) # Find nice multiple <= target, >= min_floor

         # Ensure y1 is sufficiently above the data point
         target_y1_tiny = data_max_float + step # Aim one step above
         y1_final_tiny = np.ceil(target_y1_tiny / step) * step # Find nice multiple >= target

         # Ensure y1 is strictly greater than y0
         if y1_final_tiny <= y0_final_tiny:
             y1_final_tiny = y0_final_tiny + step # Ensure at least one step range

         # Recalculate step based on the final tiny range for consistency? Or just use the step found?
         # Let's use the step calculated for the tiny range.

         return y0_final_tiny, y1_final_tiny, step


    # Calculate the step based on the meaningful range (when range > 0)
    # Use padding percentages to adjust the *total* range we want the axis to cover
    # The padding is relative to the *full_data_range*
    total_padded_range = full_data_range * (1 + (padding_percent_below + padding_percent_above) / 100.0)

    step = round_to_nice_step(total_padded_range, n_intervals=nbins)
    if step <= 0: step = 1.0 # Ensure step is positive


    # Calculate the target lower bound: data_min minus padding amount
    padding_amount_below = full_data_range * padding_percent_below / 100.0
    target_y0 = data_min_float - padding_amount_below
    # Ensure the target lower bound respects the minimum floor
    target_y0 = max(min_floor, target_y0)

    # Calculate the actual lower limit: nearest nice multiple <= target_y0
    # If target_y0 is exactly 0, np.floor(0/step)*step is 0, which is correct.
    y0_final = np.floor(target_y0 / step) * step


    # Calculate the target upper bound: data_max plus padding amount
    padding_amount_above = full_data_range * padding_percent_above / 100.0
    target_y1 = data_max_float + padding_amount_above
    # Calculate the actual upper limit: nearest nice multiple >= target_y1
    y1_final = np.ceil(target_y1 / step) * step

    # Ensure the upper limit is strictly greater than the lower limit, especially if range was tiny
    # This check might be redundant with previous logic but good as a final safeguard
    if y1_final <= y0_final:
        y1_final = y0_final + step # Add one step if limits collapsed


    return y0_final, y1_final, step

#=== Graph plotting functions ===
def graph_time(parallel_csv, sequential_csv=None, data_label="", date_prefix="", max_y=None, min_y=None,
                output_dir=None, num_log_ticks=20):
    """
    Params:
        parallel_csv (str): filepath to parallel time CSV
        sequential_csv (str, optional): filepath to sequential time CSV
        data_label (str): descriptive label for dataset
        date_prefix (str): date prefix for filenames
        max_y (float, optional): max Y-axis value; if None, computed automatically
        log_scale (bool or None): 
            - True/False forces log vs. linear scale.
            - None (default) auto-enables log if 'imp' or 'imperative' appears in parallel_csv name.
        log_base (float): base of log scale (e.g. 2, 10, math.e)
    """
    # Compute or validate max_y
    if max_y is None:
        max_y = compute_max_y([parallel_csv], [sequential_csv] if sequential_csv else None)

    # Calculate averages
    par_grouped = calculate_averages_parallel(parallel_csv)
    seq_averages = calculate_averages_seq(sequential_csv) if sequential_csv else None

    # Determine input sizes and create uniform x positions
    thread_counts = sorted(par_grouped.index.get_level_values('thread').unique())
    input_sizes = sorted(par_grouped.index.get_level_values('input').unique())
    x_positions = list(range(len(input_sizes)))

    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 7))
    for thread in thread_counts:
        thread_data = par_grouped.xs(thread, level='thread') \
                                .reindex(input_sizes).sort_index()
        color = thread_colors.get(thread, 'black')
        marker = thread_markers.get(thread, '.')
        ax.plot(
            x_positions,
            thread_data.values,
            marker=marker,
            linestyle='--',
            linewidth=2,
            markersize=8,
            label=f'{thread} Thread(s)',
            color=color
        )

    if seq_averages is not None:
        seq_vals = seq_averages.reindex(input_sizes).sort_index().values
        ax.plot(
            x_positions,
            seq_vals,
            marker='x',
            linestyle='--',
            linewidth=2,
            markersize=8,
            label='Sequential',
            color='black'
        )

    # labels & grid
    ax.set_xlabel('Input Size')
    if (log_scale):
        ax.set_ylabel('Wall Time Log Scale (ms)') 
    else:
        ax.set_ylabel('Wall Time (ms)') 
    ax.set_xticks(x_positions)
    ax.set_xticklabels(input_sizes)
    ax.set_xlim(left=min(x_positions), right=max(x_positions))
    ax.legend(title="Threads / Seq")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    #checks to see if the file is imperative. otherwise, even if log scale is enabled, it won't do it for 
    #functional paradigm plots.
    #lcname = os.path.basename(parallel_csv).lower()
    #is_imp = ("imp" in lcname) or ("imperative" in lcname)

    # y-scale: log or linear
    '''if log_scale:
        
        ax.set_yscale('log', base=log_base)

        # set exact top, let bottom auto‐adjust (must be > 0)
        ax.set_ylim(0, top=max_y * 1.05)

        # put ticks at 1,2,...,base-1 × each decade, up to max_y
        subs = list(range(1, int(log_base)))
        locator = ticker.LogLocator(base=log_base, subs=subs, numticks=50)
        ax.yaxis.set_major_locator(locator)

        # raw number formatting
        fmt = ticker.ScalarFormatter()
        fmt.set_scientific(False)
        fmt.set_useOffset(False)
        ax.yaxis.set_major_formatter(fmt)'''
    if log_scale:
        ax.set_yscale('log', base=log_base)

        #bottom_limit = max(min_y if min_y is not None else 1, 0.1)
        ax.set_ylim(min_y, top=max_y * 1.05)

        # Use LogLocator to generate "nice" tick locations on the log scale.
        # LogLocator automatically finds ticks that are powers of the base.
        # subs=[1, 2, 5] adds ticks at 1x, 2x, and 5x within each power of the base
        # (e.g., for base 10, this gives 10, 20, 50, 100, 200, 500, 1000, etc.).
        # numticks provides a hint for the maximum number of ticks to aim for.
        locator = ticker.LogLocator(base=log_base, subs=[1, 2, 5], numticks=num_log_ticks)
        ax.yaxis.set_major_locator(locator)

        # Use ScalarFormatter to display tick labels as raw numbers, not scientific notation
        fmt = ticker.ScalarFormatter()
        fmt.set_scientific(False)
        fmt.set_useOffset(False)
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.set_ylim(0, max_y * 1.05)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=20, prune='both'))

    fig.tight_layout()

    # Save plot
    filename_parts = [date_prefix, data_label, "wall_time.png"] if date_prefix else [data_label, "wall_time.png"]
    output_filename = "_".join([p for p in filename_parts if p])
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"saved time comparison plot: {output_path}")
    plt.close(fig)


'''def graph_perf_values(parallel_csv, sequential_csv=None, data_label="", date_prefix="", metrics_dict=None, max_y=None,
output_dir=None, independent_metrics=None, metrics_to_normalize_by_input=None):
    """
    Plots selected perf metrics for parallel (multi-thread) and optional sequential runs.
    
    Params:
        parallel_csv (str): path to the parallel perf CSV
        sequential_csv (str, optional): path to the sequential perf CSV
        data_label (str): descriptive label (e.g. 'spectral_norm_functional')
        date_prefix (str): prefix used in filenames (e.g. '30-04')
        metrics_dict (dict): maps metric names → bool (True to plot)
        max_y (float, optional): global Y-axis max; if None, computed per metric
        output_dir (str, optional): where to save each plot (defaults to plots_time folder)
    """

    # 1) default to global perf_metrics_collection if none passed
    if metrics_dict is None:
        metrics_dict = perf_metrics_collection

    # 2) load and sanitize
    df_par = pd.read_csv(parallel_csv)
    df_par.columns = df_par.columns.str.strip()
    df_seq = None
    if sequential_csv:
        df_seq = pd.read_csv(sequential_csv)
        df_seq.columns = df_seq.columns.str.strip()

    #normalization!
    if metrics_to_normalize_by_input:
        # Updated print statement to reflect division by input^2
        print(f"Normalizing metrics by 'input'^2 size: {metrics_to_normalize_by_input}")

        # Normalize parallel data
        if 'input' not in df_par.columns:
                print("Error: 'input' column not found in parallel CSV. Cannot normalize parallel data.")
                # Decide how to handle: continue without normalizing or return? Let's continue but skip normalization.
        else:
            # Check for division by zero before attempting division (input=0)
            if (df_par['input'] == 0).any():
                    print("Warning: Division by zero encountered in parallel data 'input' column. Skipping normalization for rows with input=0.")
                    # Create a mask to select only rows where input is not zero
                    non_zero_input_mask_par = df_par['input'] != 0
            else:
                    non_zero_input_mask_par = pd.Series(True, index=df_par.index) # All inputs are non-zero

            for metric in metrics_to_normalize_by_input:
                if metric in df_par.columns:
                    # Perform division only for rows where input is non-zero
                    # === CHANGE IS HERE: Divide by input squared ===
                    df_par.loc[non_zero_input_mask_par, metric] = df_par.loc[non_zero_input_mask_par, metric] / (df_par.loc[non_zero_input_mask_par, 'input'] ** 2)
                # else:
                    # print(f"Warning: Metric '{metric}' specified for normalization not found in parallel data.") # Optional warning

        # Normalize sequential data if it exists
        if df_seq is not None:
            if 'input' not in df_seq.columns:
                    print("Error: 'input' column not found in sequential CSV. Cannot normalize sequential data.")
                    # Continue, but sequential data won't be normalized
            else:
                # Check for division by zero before attempting division (input=0)
                if (df_seq['input'] == 0).any():
                    print("Warning: Division by zero encountered in sequential data 'input' column. Skipping normalization for rows with input=0.")
                    # Create a mask to select only rows where input is not zero
                    non_zero_input_mask_seq = df_seq['input'] != 0
                else:
                    non_zero_input_mask_seq = pd.Series(True, index=df_seq.index) # All inputs are non-zero

                for metric in metrics_to_normalize_by_input:
                    if metric in df_seq.columns:
                            # Perform division only for rows where input is non-zero
                            # === CHANGE IS HERE: Divide by input squared ===
                            df_seq.loc[non_zero_input_mask_seq, metric] = df_seq.loc[non_zero_input_mask_seq, metric] / (df_seq.loc[non_zero_input_mask_seq, 'input'] ** 2)

    # 3) pick only the metrics one has defined up in the about and actually exist
    metrics_to_plot = [
        m for m, enabled in metrics_dict.items()
        if enabled and ((m in df_par.columns) or (df_seq is not None and m in df_seq.columns))
    ]
    if not metrics_to_plot:
        print("⚠️  No perf metrics to plot for:", parallel_csv, "(or missing columns)")
        return

    # 4) build categorical x-axis (all input sizes seen in either file)
    inputs_par = set(df_par['input'].unique())
    inputs_seq = set(df_seq['input'].unique()) if df_seq is not None else set()
    input_sizes = sorted(inputs_par | inputs_seq)
    x_index = {size: i for i, size in enumerate(input_sizes)}

    # 5) know thread counts for parallel
    thread_counts = sorted(df_par['thread'].unique())

    # 6) for each metric, make one plot
    for metric in metrics_to_plot:
    # 1) if this metric should be independent, always recompute per-style
        if independent_metrics and metric in independent_metrics:
            max_y_metric = compute_max_y(
                [parallel_csv],
                [sequential_csv] if sequential_csv else None,
                metrics=[metric]
            )
        # 2) else if we got a shared dict, use that
        elif isinstance(max_y, dict):
            max_y_metric = max_y.get(metric)
        # 3) else if a single float was passed, use it
        elif isinstance(max_y, (int, float)):
            max_y_metric = max_y
        # 4) otherwise, auto‑compute across both files
        else:
            max_y_metric = compute_max_y(
                [parallel_csv],
                [sequential_csv] if sequential_csv else None,
                metrics=[metric]
            )

        fig, ax = plt.subplots(figsize=(12, 7))
        # --- parallel ---
        grp_par = df_par.groupby(['input', 'thread'])[metric].mean().reset_index()
        for thr in thread_counts:
            sub = grp_par[grp_par['thread'] == thr].sort_values('input')
            x_vals = sub['input'].map(x_index)
            ax.plot(x_vals, sub[metric],
                    marker=thread_markers.get(thr, '.'),
                    linestyle='--',
                    linewidth=2,
                    markersize=thread_markersize.get(thr, 8),
                    color=thread_colors.get(thr, 'black'),
                    label=f"{thr} threads")

        # --- sequential ---
        if df_seq is not None and metric in df_seq.columns:
            grp_seq = df_seq.groupby('input')[metric].mean().reset_index().sort_values('input')
            x_vals = grp_seq['input'].map(x_index)
            ax.plot(x_vals, grp_seq[metric],
                    marker='x', linestyle='--', linewidth=2,
                    markersize=8, color='black',
                    label="Sequential")

        # --- labels & styling ---
        ax.set_xlabel("Input Size")

        base_ylabel = metric.replace("_", " ").title()
        if metric in metrics_to_normalize_by_input:
            # Use 'per input^2' as per the normalization comment
            final_ylabel = base_ylabel + " per input^2"
        else:
            final_ylabel = base_ylabel
        ax.set_ylabel(final_ylabel)
        
        ax.set_xticks(list(x_index.values()))
        ax.set_xticklabels(input_sizes)
        ax.set_xlim(0, len(input_sizes) - 1)

        #I WANT TO ADD A WAY TO NORMALIZE MAX_Y as well
        #if metric (or is part of the metric list to normalize)
                #max_y_metric = max_y_metric / 32768 #average of input size

        ax.set_ylim(0, max_y_metric * 1.05)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10, prune='both'))
        ax.legend(title="Threads / Seq")
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()

        # --- save ---
        metric_filename_part = metric
        if metric in metrics_to_normalize_by_input:
             metric_filename_part = metric + "_per_input"

        parts = [date_prefix, data_label, metric_filename_part + ".png"] if date_prefix else [data_label, metric_filename_part + ".png"]
        filename = "_".join(p for p in parts if p)
        outdir = output_dir or plots_results_folder_time
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, filename)
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"saved perf metric plot '{metric}' → {outpath}")'''

def graph_perf_values(parallel_csv, sequential_csv=None, data_label="", date_prefix="", metrics_dict=None, max_y=None,
output_dir=None, independent_metrics=None, metrics_to_normalize_by_input=None, min_y=None):
    """
    Plots selected perf metrics for parallel (multi-thread) and optional sequential runs.

    Params:
        parallel_csv (str): path to the parallel perf CSV
        sequential_csv (str, optional): path to the sequential perf CSV
        data_label (str): descriptive label (e.g. 'spectral_norm_functional')
        date_prefix (str): prefix used in filenames (e.g. '30-04')
        metrics_dict (dict): maps metric names → bool (True to plot)
        max_y (float or dict, optional): global Y-axis max; if None, computed per metric
        min_y (float or dict, optional): global Y-axis min; if None, computed per metric
        output_dir (str, optional): where to save each plot (defaults to plots_time folder)
    """

    # 1) default to global perf_metrics_collection if none passed
    if metrics_dict is None:
        metrics_dict = perf_metrics_collection

    # 2) load and sanitize
    try:
        df_par = pd.read_csv(parallel_csv)
        df_par.columns = df_par.columns.str.strip()
    except Exception as e:
        print(f"Error reading parallel perf CSV {parallel_csv}: {e}")
        return # Cannot proceed without parallel data

    df_seq = None
    if sequential_csv:
        try:
            df_seq = pd.read_csv(sequential_csv)
            df_seq.columns = df_seq.columns.str.strip()
        except Exception as e:
            print(f"Error reading sequential perf CSV {sequential_csv}: {e}")
            df_seq = None # Continue without sequential data


    #normalization!
    if metrics_to_normalize_by_input:
        # Updated print statement to reflect division by input^2
        print(f"Normalizing metrics by 'input'^2 size: {metrics_to_normalize_by_input}")

        # Normalize parallel data
        if 'input' not in df_par.columns:
                print("Error: 'input' column not found in parallel CSV. Cannot normalize parallel data.")
                # Decide how to handle: continue without normalizing or return? Let's continue but skip normalization.
        else:
            # Check for division by zero before attempting division (input=0)
            if (df_par['input'] == 0).any():
                    print("Warning: Division by zero encountered in parallel data 'input' column. Skipping normalization for rows with input=0.")
                    # Create a mask to select only rows where input is not zero
                    non_zero_input_mask_par = df_par['input'] != 0
            else:
                    non_zero_input_mask_par = pd.Series(True, index=df_par.index) # All inputs are non-zero

            for metric in metrics_to_normalize_by_input:
                if metric in df_par.columns and pd.api.types.is_numeric_dtype(df_par[metric]):
                    # Perform division only for rows where input is non-zero
                    # === CHANGE IS HERE: Divide by input squared ===
                    df_par.loc[non_zero_input_mask_par, metric] = df_par.loc[non_zero_input_mask_par, metric] / (df_par.loc[non_zero_input_mask_par, 'input'] ** 2)
                # else:
                    # print(f"Warning: Metric '{metric}' specified for normalization not found or not numeric in parallel data.") # Optional warning

        # Normalize sequential data if it exists
        if df_seq is not None:
            if 'input' not in df_seq.columns:
                    print("Error: 'input' column not found in sequential CSV. Cannot normalize sequential data.")
                    # Continue, but sequential data won't be normalized
            else:
                # Check for division by zero before attempting division (input=0)
                if (df_seq['input'] == 0).any():
                    print("Warning: Division by zero encountered in sequential data 'input' column. Skipping normalization for rows with input=0.")
                    # Create a mask to select only rows where input is non-zero
                    non_zero_input_mask_seq = df_seq['input'] != 0
                else:
                    non_zero_input_mask_seq = pd.Series(True, index=df_seq.index) # All inputs are non-zero

                for metric in metrics_to_normalize_by_input:
                     if metric in df_seq.columns and pd.api.types.is_numeric_dtype(df_seq[metric]):
                            # Perform division only for rows where input is non-zero
                            # === CHANGE IS HERE: Divide by input squared ===
                            df_seq.loc[non_zero_input_mask_seq, metric] = df_seq.loc[non_zero_input_mask_seq, metric] / (df_seq.loc[non_zero_input_mask_seq, 'input'] ** 2)
                     # else:
                         # print(f"Warning: Metric '{metric}' specified for normalization not found or not numeric in sequential data.") # Optional warning


    # 3) pick only the metrics one has defined up in the about and actually exist
    metrics_to_plot = [
        m for m, enabled in metrics_dict.items()
        if enabled and ((m in df_par.columns) or (df_seq is not None and m in df_seq.columns))
    ]
    if not metrics_to_plot:
        print("⚠️  No perf metrics to plot for:", parallel_csv, "(or missing columns)")
        return

    # 4) build categorical x-axis (all input sizes seen in either file)
    inputs_par = set(df_par['input'].unique())
    inputs_seq = set(df_seq['input'].unique()) if df_seq is not None else set()
    input_sizes = sorted(inputs_par | inputs_seq)
    x_index = {size: i for i, size in enumerate(input_sizes)}

    # 5) know thread counts for parallel
    # Ensure 'thread' column exists before trying to get unique values
    thread_counts = sorted(df_par['thread'].unique()) if 'thread' in df_par.columns else []


    # 6) for each metric, make one plot
    for metric in metrics_to_plot:
        # Determine max_y_metric
        max_y_metric_val = None # Initialize

        if independent_metrics and metric in independent_metrics:
            # For independent metrics, compute max locally
            local_max_result = compute_max_or_min_y_norm(
                [parallel_csv],
                [sequential_csv] if sequential_csv else None,
                metrics=[metric],
                metrics_to_normalize=metrics_to_normalize_by_input,
                aggregate_type='max'
            )
            # Safely get the scalar value from local_max_result
            if isinstance(local_max_result, dict):
                 max_y_metric_val = local_max_result.get(metric)
            elif isinstance(local_max_result, (int, float)) or local_max_result is None:
                 max_y_metric_val = local_max_result


        # If max_y_metric_val is still None (or invalid), try the externally provided max_y
        if max_y_metric_val is None or not isinstance(max_y_metric_val, (int, float)) or np.isnan(max_y_metric_val) or np.isinf(max_y_metric_val):
             if isinstance(max_y, dict):
                 max_y_metric_val = max_y.get(metric)
             elif isinstance(max_y, (int, float)):
                 max_y_metric_val = max_y
             # If external max_y is also not suitable, max_y_metric_val remains None


        # --- Determine min_y_metric - Use external min_y if provided, otherwise compute ---
        min_y_metric_val = None # Initialize

        if isinstance(min_y, dict):
            # If min_y is a dict, try to get the metric-specific min
            min_y_metric_val = min_y.get(metric)
        elif isinstance(min_y, (int, float)):
            # If min_y is a scalar, use that scalar for this metric
            min_y_metric_val = min_y
        # If min_y is None or not a valid type, min_y_metric_val remains None


        # If min_y_metric_val is still None (or invalid), compute it internally
        if min_y_metric_val is None or not isinstance(min_y_metric_val, (int, float)) or np.isnan(min_y_metric_val) or np.isinf(min_y_metric_val):
             # Compute internally using the new function for min
             computed_min_result = compute_max_or_min_y_norm(
                  [parallel_csv],
                  [sequential_csv] if sequential_csv else None,
                  metrics=[metric],
                  metrics_to_normalize=metrics_to_normalize_by_input,
                  aggregate_type='min'
             )
             # Safely get the scalar value from computed_min_result (could be scalar, dict, or None)
             if isinstance(computed_min_result, dict):
                 min_y_metric_val = computed_min_result.get(metric)
             elif isinstance(computed_min_result, (int, float)) or computed_min_result is None:
                 min_y_metric_val = computed_min_result
             # Else: computed_min_result is some unexpected type, min_y_metric_val remains None


        # Use find_nice_axis_limits to get rounded, padded limits and step
        # Pass the determined *scalar* min_y_metric_val and max_y_metric_val
        # find_nice_axis_limits will handle the None/NaN/Inf case with a default range
        # Note: The min_floor is set to 0.0 explicitly here.
        y0_final, y1_final, step = find_nice_axis_limits(min_y_metric_val, max_y_metric_val, nbins=10, min_floor=-0.25, padding_percent_below=15.0, padding_percent_above=5.0)


        # Add a print for debugging/info
        # print(f"Metric: '{metric}' - Data range basis [{min_y_metric_val}, {max_y_metric_val}] -> Axis limits [{y0_final:.2f}, {y1_final:.2f}] with step {step:.2f}")


        fig, ax = plt.subplots(figsize=(12, 7))
        # --- parallel ---
        # Ensure only columns needed for groupby and the metric are selected for performance/safety
        if metric in df_par.columns and not df_par.empty: # Only attempt to plot if metric exists and df is not empty
             # Need to align parallel data with all input sizes for consistent x-axis
             grp_par = df_par.groupby(['input', 'thread'])[metric].mean().reset_index()
             # Create a DataFrame with all possible input/thread combinations
             all_inputs_threads = pd.MultiIndex.from_product([input_sizes, thread_counts], names=['input', 'thread']).to_frame(index=False)
             # Merge the grouped data with the full set of inputs/threads to include missing inputs
             # Use pd.concat if grp_par might be empty, then groupby again if needed? No, merge should handle empty grp_par fine (resulting in all NaNs).
             merged_par = pd.merge(all_inputs_threads, grp_par, on=['input', 'thread'], how='left')

             for thr in thread_counts:
                 # Filter the merged data for this thread
                 sub = merged_par[merged_par['thread'] == thr].sort_values('input')
                 x_vals = sub['input'].map(x_index)
                 ax.plot(x_vals, sub[metric],
                         marker=thread_markers.get(thr, '.'),
                         linestyle='--',
                         linewidth=2,
                         markersize=thread_markersize.get(thr, 8),
                         color=thread_colors.get(thr, 'black'),
                         label=f"{thr} threads")
        elif metric not in df_par.columns:
             print(f"Warning: Metric '{metric}' not found in parallel data for plotting.")


        # --- sequential ---
        if df_seq is not None and metric in df_seq.columns and not df_seq.empty: # Only attempt to plot if sequential data exists, metric is in columns, and df is not empty
            # Need to align sequential data with all input sizes
            grp_seq = df_seq.groupby('input')[metric].mean().reset_index()
            # Create a DataFrame with all possible input sizes
            all_inputs = pd.DataFrame({'input': input_sizes})
            # Merge the grouped data with the full set of inputs to include missing inputs
            merged_seq = pd.merge(all_inputs, grp_seq, on='input', how='left').sort_values('input')

            x_vals = merged_seq['input'].map(x_index)
            ax.plot(x_vals, merged_seq[metric],
                    marker='x', linestyle='--', linewidth=2,
                    markersize=8, color='black',
                    label="Sequential")
        elif df_seq is not None and metric not in df_seq.columns: # Sequential data exists, but metric is missing
             print(f"Warning: Metric '{metric}' not found in sequential data for plotting.")


        # --- labels & styling ---
        ax.set_xlabel("Input Size")

        base_ylabel = metric.replace("_", " ").title()
        if metric in metrics_to_normalize_by_input:
            # Use 'per input^2' as per the normalization comment
            final_ylabel = base_ylabel + " per input^2"
        else:
            final_ylabel = base_ylabel
        ax.set_ylabel(final_ylabel)

        # Ensure x_index is not empty before setting x-ticks and x-limits
        if x_index:
             ax.set_xticks(list(x_index.values()))
             ax.set_xticklabels(input_sizes)
             ax.set_xlim(left=min(x_index.values()), right=max(x_index.values()))
        else:
             # If no input sizes found, set empty ticks/labels and a default x-range
             ax.set_xticks([])
             ax.set_xticklabels([])
             ax.set_xlim(0, 1) # Default tiny range


        # Set y-axis limits using the nice limits calculated
        ax.set_ylim(y0_final, y1_final)

        # Use MultipleLocator to place ticks at multiples of the nice step size
        ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
        # Use ScalarFormatter for y-axis ticks
        fmt = ticker.ScalarFormatter(useOffset=False, useMathText=False)
        ax.yaxis.set_major_formatter(fmt)

        ax.legend(title="Threads / Seq")
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()

        # --- save ---
        metric_filename_part = metric
        if metric in metrics_to_normalize_by_input:
             metric_filename_part = metric + "_per_input"

        parts = [date_prefix, data_label, metric_filename_part + ".png"] if date_prefix else [data_label, metric_filename_part + ".png"]
        filename = "_".join(p for p in parts if p)
        outdir = output_dir or plots_results_folder_perf
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, filename)
        try:
            plt.savefig(outpath, dpi=300, bbox_inches='tight')
            print(f"saved perf metric plot '{metric}' → {outpath}")
        except Exception as e:
             print(f"Error saving plot {outpath}: {e}")

        plt.close(fig)

def graph_sys_mem(parallel_csv, sequential_csv=None, data_label="", date_prefix="", metrics_dict=None, max_y=None, min_y=None,
output_dir=None, baseline_mem=None):
    '''
    Params:
    (all optional) csv_imp/func_seq, csv_imp/func_parallel
    Returns:
    - Two graphs of system memory usage from running the programs. One showing imperative results, 
    the other functional results.
    '''
    # 1) default to global mem_metrics_collection if none passed
    if metrics_dict is None:
        metrics_dict = mem_metrics_collection

    # 2) load and sanitize
    #parallel csv
    df_par = pd.read_csv(parallel_csv)
    df_par.columns = df_par.columns.str.strip()

    #converts kb to mb
    if 'free_mem' in df_par.columns:
        if baseline_subtract_true:
            df_par['free_mem'] = baseline_mem - df_par['free_mem']
        df_par['free_mem'] = df_par['free_mem'] / 1024.0
    if 'pid_mem' in df_par.columns:
        df_par['pid_mem'] = df_par['pid_mem'] / 1024.0

    #sequential csv - defaults to none if there is not one to begin with.
    df_seq = None
    if sequential_csv:
        df_seq = pd.read_csv(sequential_csv)
        df_seq.columns = df_seq.columns.str.strip()

        if baseline_subtract_true:
            df_seq['free_mem'] = baseline_mem - df_seq['free_mem']
            #converts kb to mb
        if 'free_mem' in df_seq.columns:
            df_seq['free_mem'] = df_seq['free_mem'] / 1024.0
        if 'pid_mem' in df_seq.columns:
            df_seq['pid_mem'] = df_seq['pid_mem'] / 1024.0

    # 3) pick only the metrics one has defined up in the top and those that actually exist
    metrics_to_plot = [
        m for m, enabled in metrics_dict.items()
        if enabled and ((m in df_par.columns) or (df_seq is not None and m in df_seq.columns))
    ]
    if not metrics_to_plot:
        print("⚠️  No perf metrics to plot for:", parallel_csv, "(or missing columns)")
        return

    # 4) build categorical x-axis (all input sizes seen in either file)
    inputs_par = set(df_par['input'].unique())
    inputs_seq = set(df_seq['input'].unique()) if df_seq is not None else set()
    input_sizes = sorted(inputs_par | inputs_seq)
    x_index = {size: i for i, size in enumerate(input_sizes)}

    # 5) know thread counts for parallel
    thread_counts = sorted(df_par['thread'].unique())

    # 6) for each metric, make one plot
    for metric in metrics_to_plot:
    # Decide y‐limits
        # min
        if isinstance(min_y, dict):
            y0 = min_y.get(metric, 0)
            if metric == "free_mem" or "pid_mem":

                if baseline_subtract_true and metric == "free_mem":
                    y0 = baseline_mem - y0

                y0 = y0 / 1024.0
        elif isinstance(min_y, (int,float)):
            y0 = min_y
        else:
            # auto‐compute
            mins = compute_min_y([parallel_csv],
                                 [sequential_csv] if sequential_csv else None,
                                 metrics=[metric])
            y0 = mins if isinstance(mins, (int,float)) else mins.get(metric, 0)


        # max
        if isinstance(max_y, dict):
            y1 = max_y.get(metric, None)
            if metric == "free_mem" or "pid_mem":

                if baseline_subtract_true and metric == "free_mem":
                    if y1 is not None: # Only adjust if y1 is not None
                        y1 = baseline_mem - y1

                y1 = y1 / 1024.0
        elif isinstance(max_y, (int,float)):
            y1 = max_y
        else:
            y1 = compute_max_y([parallel_csv],
                               [sequential_csv] if sequential_csv else None,
                               metrics=[metric])

        fig, ax = plt.subplots(figsize=(12, 7))
        # --- parallel ---
        grp_par = df_par.groupby(['input', 'thread'])[metric].mean().reset_index()
        for thr in thread_counts:
            sub = grp_par[grp_par['thread'] == thr].sort_values('input')
            x_vals = sub['input'].map(x_index)
            ax.plot(x_vals, sub[metric],
                    marker=thread_markers.get(thr, '.'),
                    linestyle='--', linewidth=2,
                    markersize=thread_markersize.get(thr, 8),
                    color=thread_colors.get(thr, 'black'),
                    label=f"{thr} threads")

        # --- sequential ---
        if df_seq is not None and metric in df_seq.columns:
            grp_seq = df_seq.groupby('input')[metric].mean().reset_index().sort_values('input')
            x_vals = grp_seq['input'].map(x_index)
            ax.plot(x_vals, grp_seq[metric],
                    marker='x', linestyle='--', linewidth=2,
                    markersize=8, color='black',
                    label="Sequential")

        # --- labels & styling ---
        ax.set_xlabel("Input Size")

        ylabel_text = "RSS"
        if metric == "free_mem" and baseline_subtract_true:
             ylabel_text = "Memory Used (Baseline - Free Mem)" # Or similar descriptive label
        ylabel_text += " (MB)" # Add units
        ax.set_ylabel(ylabel_text)
        ax.set_xticks(list(x_index.values()))
        ax.set_xticklabels(input_sizes)
        ax.set_xlim(0, len(input_sizes) - 1)

        if metric == "free_mem" and baseline_subtract_true:
            if selected_subfolder == "mandelbrot":
                ax.set_ylim(y1 * -1.001, y0)
            else:
                ax.set_ylim(y1 * 0.9, y0)
        else:
            if selected_subfolder == "mandelbrot":
                ax.set_ylim(y0 * -1.001, y1)
            else:
                ax.set_ylim(y0 * 0.5, y1)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10, prune='both'))
        ax.legend(title="Threads / Seq")
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()

        # --- save ---
        parts = [date_prefix, data_label, metric + ".png"] if date_prefix else [data_label, metric + ".png"]
        filename = "_".join(p for p in parts if p)
        outdir = output_dir or plots_results_folder_time
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, filename)
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"saved system memory metric plot '{metric}' → {outpath}")

def graph_pid_mem_peak(parallel_csv, sequential_csv=None, data_label="", date_prefix="", output_dir=None, max_y=None, min_y=None):
    '''
    Plots the PEAK 'pid_mem' metric (Process Memory) versus input size.
    Calculates the maximum pid_mem value for each (input, thread) or input group.
    Handles KB to MB conversion and uses nice axis limits starting from 0 or minimum data.
    Accepts optional pre-calculated max_y/min_y for shared scaling.

    Params:
    parallel_csv (str): path to the parallel memory CSV
    sequential_csv (str, optional): path to the sequential memory CSV
    data_label (str): descriptive label (e.g. 'spectral_norm_imperative')
    date_prefix (str): prefix used in filenames (e.g. '30-04')
    output_dir (str, optional): where to save each plot (defaults to plots_mem folder)
    max_y (float or dict, optional): Optional pre-calculated max y-limit. If a dict, assumes {metric: max_val}.
    min_y (float or dict, optional): Optional pre-calculated min y-limit. If a dict, assumes {metric: min_val}.
    '''
    metric = 'pid_mem' # This function is hardcoded for pid_mem

    # 1) load data (Simplified: Removed try/except)
    df_par = pd.read_csv(parallel_csv)
    df_par.columns = df_par.columns.str.strip()
    # Simplified: Assuming metric column exists and is numeric
    df_par[metric] = df_par[metric] / 1024.0 # Apply KB to MB conversion

    df_seq = None
    if sequential_csv:
        # Simplified: Removed try/except
        df_seq = pd.read_csv(sequential_csv)
        df_seq.columns = df_seq.columns.str.strip()
        # Simplified: Assuming metric column exists and is numeric
        df_seq[metric] = df_seq[metric] / 1024.0 # Apply KB to MB conversion


    # 1) Calculate Peak data (group and take max) (Simplified checks)
    grp_par_peaks = pd.DataFrame()
    # Simplified: Assuming input and thread columns exist for parallel
    grp_par_peaks = df_par.groupby(['input', 'thread'])[metric].max().reset_index()


    grp_seq_peaks = pd.DataFrame()
    # Simplified: Assuming input column exists for sequential if df_seq is not None
    if df_seq is not None:
         grp_seq_peaks = df_seq.groupby('input')[metric].max().reset_index()


    # 2) Determine the range of the calculated peak data
    all_peak_values = []
    # Simplified: dropna is used, assuming data is clean enough
    if not grp_par_peaks.empty:
        all_peak_values.extend(grp_par_peaks[metric].dropna().tolist())
    if not grp_seq_peaks.empty: # Keep check for empty sequential peaks
        all_peak_values.extend(grp_seq_peaks[metric].dropna().tolist())

    # Simplified: Assuming all_peak_values is not empty
    peak_data_min = np.min(all_peak_values)
    peak_data_max = np.max(all_peak_values)


    # --- Determine y-axis limits ---
    y0_final = 0.0
    y1_final = 1.0 # Default small range in case calculation fails
    step = 1.0   # Default step

    # Check if max_y and min_y were provided externally
    if max_y is not None and min_y is not None:
         # Use provided limits (handle if they are in a dict)
         provided_min = min_y.get(metric) if isinstance(min_y, dict) else min_y
         provided_max = max_y.get(metric) if isinstance(max_y, dict) else max_y

         # Simplified: Assume provided_min and provided_max are valid numbers
         y0_final = max(0.0, provided_min) # Ensure lower bound is >= 0
         y1_final = provided_max # Use provided max as the upper limit

         # Calculate step based on the provided range
         provided_range = y1_final - y0_final
         step = round_to_nice_step(provided_range, n_intervals=10)
         if step <= 0: step = 1.0 # Ensure step is positive

         # No need to add padding to y1_final if limits are explicitly set?
         # Or add padding *after* using provided max? Let's add padding consistent with the other case.
         y1_final = y1_final * 1.05 # Add consistent padding to the upper limit

         # Ensure y1_final is still > y0_final after padding/rounding quirks
         if y1_final <= y0_final: y1_final = y0_final + step # Add a step if limits collapsed

         print(f"Using provided Y limits for '{metric}': [{y0_final:.2f}, {y1_final:.2f}] with step {step:.2f}")

    else:
        # Use the calculated peak data range to find nice limits
        # Simplified: Using the simplified find_nice_axis_limits
        # find_nice_axis_limits already includes padding on y1
        y0_nice, y1_nice_padded, step = find_nice_axis_limits(peak_data_min, peak_data_max, nbins=10, min_floor=0.0)
        y0_final = y0_nice
        y1_final = y1_nice_padded
        print(f"Using calculated Y limits for '{metric}': [{y0_final:.2f}, {y1_final:.2f}] with step {step:.2f}")


    # 4) build categorical x-axis (all input sizes seen in the peak dataframes)
    # Use input sizes from the calculated peak dataframes
    inputs_par_peaks = set(grp_par_peaks['input'].unique());
    inputs_seq_peaks = set(grp_seq_peaks['input'].unique()) if not grp_seq_peaks.empty else set()
    input_sizes = sorted(list(inputs_par_peaks | inputs_seq_peaks))
    x_index = {size: i for i, size in enumerate(input_sizes)}

    # Simplified: Assuming input sizes are found
    # Simplified: Assuming thread counts are found from parallel peaks
    thread_counts = sorted(grp_par_peaks['thread'].unique()) if not grp_par_peaks.empty else []


    # 6) Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # --- parallel ---
    if not grp_par_peaks.empty: # Plot if parallel peaks were calculated
        for thr in thread_counts:
            # Filter for this thread from the peak dataframe
            thread_peaks_df = grp_par_peaks[grp_par_peaks['thread'] == thr]
            # --- FIX START: Use merge to align on 'input' column ---
            all_inputs_df = pd.DataFrame({'input': input_sizes})
            sub = pd.merge(all_inputs_df, thread_peaks_df, on='input', how='left').sort_values('input')
            # --- FIX END ---

            x_vals = sub['input'].map(x_index)
            # Simplified: Assuming valid data for plotting lines
            ax.plot(x_vals, sub[metric],
                    marker=thread_markers.get(thr, '.'),
                    linestyle='--', linewidth=2,
                    markersize=thread_markersize.get(thr, 8),
                    color=thread_colors.get(thr, 'black'),
                    label=f"{thr} threads")


    # --- sequential ---
    if not grp_seq_peaks.empty: # Plot if sequential peaks were calculated
        # --- FIX START: Use merge to align on 'input' column ---
        all_inputs_df = pd.DataFrame({'input': input_sizes})
        grp_seq_peaks_reindexed = pd.merge(all_inputs_df, grp_seq_peaks, on='input', how='left').sort_values('input')
        # --- FIX END ---

        x_vals = grp_seq_peaks_reindexed['input'].map(x_index)
        # Simplified: Assuming valid data for plotting lines
        ax.plot(x_vals, grp_seq_peaks_reindexed[metric],
                marker='x', linestyle='--', linewidth=2,
                markersize=8, color='black',
                label="Sequential")


    # --- labels & styling ---
    ax.set_xlabel("Input Size")
    ax.set_ylabel("Peak Process Memory (MB)") # Updated label to reflect "Peak"

    ax.set_xticks(list(x_index.values())); ax.set_xticklabels(input_sizes)
    # Simplified: Assuming x_index is not empty
    ax.set_xlim(left=min(x_index.values()), right=max(x_index.values()));

    # Set y-axis limits using the nice limits calculated from peak data range OR provided limits
    ax.set_ylim(y0_final, y1_final) # Use the determined final limits

    # Use MultipleLocator to place ticks at multiples of the nice step size
    ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
    # Use ScalarFormatter for y-axis ticks
    fmt = ticker.ScalarFormatter(useOffset=False, useMathText=False)
    ax.yaxis.set_major_formatter(fmt)


    ax.legend(title="Threads / Seq")
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()

    # --- save ---
    parts = [date_prefix, data_label, "peak_" + metric + ".png"] if date_prefix else [data_label, "peak_" + metric + ".png"] # Added "peak_" to filename
    filename = "_".join(p for p in parts if p)
    outdir = output_dir or os.path.join(base_dir, PLOTS_FOLDER_MAP["mem"]) # Use plots_mem folder
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, filename);
    # Simplified: Removed error handling for saving
    plt.savefig(outpath, dpi=300, bbox_inches='tight'); print(f"saved peak process memory plot '{metric}' → {outpath}")

    plt.close(fig)

#=== Main ===
#convert all perf .txt/.data files to CSV
#convert every perf .data or .txt in this subfolder (parallel + sequential)
for fname in os.listdir(perf_folder):
    in_path = os.path.join(perf_folder, fname)
    # skip directories and already‑CSV files
    if not os.path.isfile(in_path) or fname.lower().endswith('.csv'):
        continue

    out_name = os.path.splitext(fname)[0] + '.csv'
    out_path = os.path.join(perf_folder, out_name)
    print(f"[parse_perf_data] {fname} → {out_name}")
    parse_perf_data(in_path, out_path)

# Load all files from each program type folder
time_files_all = list_files(time_folder, extensions=['.csv'])
perf_files_all = list_files(perf_folder, extensions=['.data', '.csv'])
mem_files_all = list_files(mem_folder, extensions=['.csv'])
#mem_pid_files_all = list_files(pidmem_folder, extensions=['.csv'])


# Parallel files (required)
time_files = filter_files(time_files_all, parallel_date, "parallel")
perf_files = filter_files(perf_files_all, parallel_date, "parallel")
mem_files = filter_files(mem_files_all, parallel_date, "parallel")
#mem_pid_files = filter_files(mem_pid_files_all, parallel_date, "parallel")

# Sequential files (optional)
if sequential_date:
    time_files_seq = filter_files(time_files_all, sequential_date, "sequential")
    perf_files_seq = filter_files(perf_files_all, sequential_date, "sequential")
    mem_files_seq = filter_files(mem_files_all, sequential_date, "sequential")
    #mem_pid_files_seq = filter_files(mem_pid_files_all, sequential_date, "sequential")
else:
    time_files_seq = []
    perf_files_seq = []
    mem_files_seq = []
    #mem_pid_files_seq = []

# Split parallel files by implementation style
time_files_by_style = split_by_style(time_files)
perf_files_by_style = split_by_style(perf_files)
mem_files_by_style = split_by_style(mem_files)
#mem_pid_files_by_style = split_by_style(mem_pid_files)

# Split sequential files by implementation style
time_files_seq_by_style = split_by_style(time_files_seq)
perf_files_seq_by_style = split_by_style(perf_files_seq)
mem_files_seq_by_style = split_by_style(mem_files_seq)
#mem_pid_files_seq_by_style = split_by_style(mem_pid_files_seq)


# === Plotting wall time versus inputsize ===
plots_results_folder_time = os.path.join(base_dir, PLOTS_FOLDER_MAP["time"])
os.makedirs(plots_results_folder_time, exist_ok=True)

#imperative plot
if time_files_by_style['imperative']:
    imp_par_list = time_files_by_style['imperative']
    imp_seq_list = time_files_seq_by_style.get('imperative', [])

imp_par_list  = time_files_by_style['imperative']
imp_seq_list  = time_files_seq_by_style.get('imperative', [])
func_par_list = time_files_by_style['functional']
func_seq_list = time_files_seq_by_style.get('functional', [])

#determine whether to use a common max_y or keep the max_y seperate in case of missing second parallel results
#e.g func parallel or imp parallel is missing.
if imp_par_list and func_par_list:
    # both parallel CSVs present -> use the common scale
    combined_par = imp_par_list + func_par_list
    combined_seq = imp_seq_list + func_seq_list
    common_max_y = compute_max_y(combined_par, combined_seq, metrics=['avg_time'])
    #common_min_y = compute_min_y(combined_par, combined_seq, metrics=['avg_time'])
    max_y_imp  = max_y_func = common_max_y
    #min_y_imp = min_y_func = common_min_y
else:
    # else, fall back to per‐style scaling
    max_y_imp  = compute_max_y(imp_par_list,  imp_seq_list,  metrics=['avg_time']) if imp_par_list else None
    #min_y_imp  = compute_min_y(imp_par_list,  imp_seq_list,  metrics=['avg_time']) if imp_par_list else None
    max_y_func = compute_max_y(func_par_list, func_seq_list, metrics=['avg_time']) if func_par_list else None
    #min_y_func = compute_min_y(func_par_list, func_seq_list, metrics=['avg_time']) if func_par_list else None

# Imperative plot
if imp_par_list:
    graph_time(
        parallel_csv   = imp_par_list[0],
        sequential_csv = imp_seq_list[0] if imp_seq_list else None,
        data_label     = f"{selected_subfolder}_imperative",
        date_prefix    = parallel_date,
        max_y          = max_y_imp,
        min_y          = 20 if (selected_subfolder == "mandelbrot") else 200,
        output_dir     = plots_results_folder_time
    )

# Functional plot
if func_par_list:
    graph_time(
        parallel_csv   = func_par_list[0],
        sequential_csv = func_seq_list[0] if func_seq_list else None,
        data_label     = f"{selected_subfolder}_functional",
        date_prefix    = parallel_date,
        max_y          = max_y_func,
        min_y          = 20 if (selected_subfolder == "mandelbrot") else 200,
        output_dir     = plots_results_folder_time
    )


# === Plotting perf values versus inputsize ===
# prepare folders
plots_results_folder_perf = os.path.join(base_dir, PLOTS_FOLDER_MAP["perf"])
os.makedirs(plots_results_folder_perf, exist_ok=True)

# Lists for each style
imp_par_list  = perf_files_by_style['imperative']
imp_seq_list  = perf_files_seq_by_style.get('imperative', [])
func_par_list = perf_files_by_style['functional']
func_seq_list = perf_files_seq_by_style.get('functional', [])

#which perf metrics are we plotting?
metrics_list = [m for m, flag in perf_metrics_collection.items() if flag]

#decide on common vs. per‑style
if imp_par_list and func_par_list:
    combined_par = imp_par_list + func_par_list
    combined_seq = imp_seq_list + func_seq_list
    # compute_max_y with multiple metrics returns a dict {metric: max_val}
    max_y_perf = compute_max_or_min_y_norm(combined_par, combined_seq, 
                                           metrics=metrics_list, 
                                           metrics_to_normalize = metrics_to_divide_with_input_size,
                                           aggregate_type='max')
    min_y_perf = compute_max_or_min_y_norm(combined_par, combined_seq, 
                                           metrics=metrics_list, 
                                           metrics_to_normalize = metrics_to_divide_with_input_size,
                                           aggregate_type='min')
else:
    # fallback to per‐style: here we only need two separate dicts
    max_y_imp  = compute_max_or_min_y_norm(imp_par_list,  imp_seq_list,  metrics=metrics_list, metrics_to_normalize = metrics_to_divide_with_input_size, aggregate_type='max') if imp_par_list else {}
    max_y_func = compute_max_or_min_y_norm(func_par_list, func_seq_list, metrics=metrics_list, metrics_to_normalize = metrics_to_divide_with_input_size, aggregate_type='max') if func_par_list else {}
    # Then pick the right dict for each style:
    #   for imp call → max_y_imp
    #   for func call → max_y_func

# imperative
if perf_files_by_style['imperative']:
    graph_perf_values(
        parallel_csv = perf_files_by_style['imperative'][0],
        sequential_csv = perf_files_seq_by_style.get('imperative', [None])[0],
        data_label = f"{selected_subfolder}_imp",
        date_prefix = parallel_date,
        max_y = max_y_perf if (imp_par_list and func_par_list) else max_y_imp,
        output_dir = plots_results_folder_perf,
        independent_metrics = indep,
        metrics_to_normalize_by_input = metrics_to_divide_with_input_size,
        min_y = min_y_perf
    )

# functional
if perf_files_by_style['functional']:
    graph_perf_values(
        parallel_csv = perf_files_by_style['functional'][0],
        sequential_csv = perf_files_seq_by_style.get('functional', [None])[0],
        data_label = f"{selected_subfolder}_func",
        max_y = max_y_perf if (imp_par_list and func_par_list) else max_y_imp,
        date_prefix = parallel_date,
        output_dir = plots_results_folder_perf,
        independent_metrics = indep,
        metrics_to_normalize_by_input = metrics_to_divide_with_input_size,
        min_y = min_y_perf
    )

#=== Plotting sys mem versus inputsize ===
add_missing_column_names_and_clean(mem_files_all)
plots_results_folder_mem = os.path.join(base_dir, PLOTS_FOLDER_MAP["mem"])
os.makedirs(plots_results_folder_mem, exist_ok=True)

# Lists for each style
imp_par_list  = mem_files_by_style['imperative']
imp_seq_list  = mem_files_seq_by_style.get('imperative', [])
func_par_list = mem_files_by_style['functional']
func_seq_list = mem_files_seq_by_style.get('functional', [])

combined_par = None
combined_seq = None

#which memory metrics are we plotting?
metrics_list = [m for m, flag in mem_metrics_collection.items() if flag]

#decide on common vs. per‑style
if imp_par_list and func_par_list:
    combined_par = imp_par_list + func_par_list
    combined_seq = imp_seq_list + func_seq_list
    # compute_max_y with multiple metrics returns a dict {metric: max_val}
    max_y_mem = compute_max_y(combined_par, combined_seq, metrics=metrics_list)
    min_y_mem = compute_min_y(combined_par, combined_seq, metrics=metrics_list)
else:
    # fallback to per‐style: here we only need two separate dicts
    max_y_imp  = compute_max_y(imp_par_list,  imp_seq_list,  metrics=metrics_list) if imp_par_list else {}
    max_y_func = compute_max_y(func_par_list, func_seq_list, metrics=metrics_list) if func_par_list else {}

    min_y_imp  = compute_min_y(imp_par_list,  imp_seq_list,  metrics=metrics_list) if imp_par_list else {}
    min_y_func = compute_min_y(func_par_list, func_seq_list, metrics=metrics_list) if func_par_list else {}
    # Then pick the right dict for each style:
    #   for imp call → max_y_imp and min_y_imp
    #   for func call → max_y_func and min_y_func

imp_seq = imp_seq_list[0] if imp_seq_list else None
func_seq = func_seq_list[0] if func_seq_list else None

# imperative
if mem_files_by_style['imperative']:
    graph_sys_mem(
        parallel_csv = mem_files_by_style['imperative'][0],
        sequential_csv =imp_seq, # error here! index out of range???
        data_label = f"{selected_subfolder}_imp",
        date_prefix = parallel_date,
        max_y = max_y_mem if (imp_par_list and func_par_list) else max_y_imp,
        min_y = min_y_mem if (imp_par_list and func_par_list) else min_y_imp,
        output_dir = plots_results_folder_mem,
        baseline_mem = baseline_mem_mandelbrot if (selected_subfolder == "mandelbrot") else baseline_mem_spec_19_05
    )

# functional
if mem_files_by_style['functional']:
    graph_sys_mem(
        parallel_csv = mem_files_by_style['functional'][0],
        sequential_csv = func_seq,
        data_label = f"{selected_subfolder}_func",
        max_y = max_y_mem if (imp_par_list and func_par_list) else max_y_imp,
        min_y = min_y_mem if (imp_par_list and func_par_list) else min_y_imp,
        date_prefix = parallel_date,
        output_dir = plots_results_folder_mem,
        baseline_mem = baseline_mem_mandelbrot if (selected_subfolder == "mandelbrot") else baseline_mem_spec_19_05
    )

#=== Plotting PID mem (Process Memory) versus inputsize ===
print("\nPlotting Process Memory (PID)...")
# Output folder is the same as system memory
plots_results_folder_mem = os.path.join(base_dir, PLOTS_FOLDER_MAP["mem"])
os.makedirs(plots_results_folder_mem, exist_ok=True)

peak_mem_metrics_list = ['pid_mem']

common_max_peak_mem = compute_peak_max_y(combined_par, combined_seq, metrics=peak_mem_metrics_list)
common_min_peak_mem = compute_peak_min_y(combined_par, combined_seq, metrics=peak_mem_metrics_list)

# imperative
if mem_files_by_style['imperative']:
    graph_pid_mem_peak(
        parallel_csv = mem_files_by_style['imperative'][0],
        sequential_csv = imp_seq,
        data_label = f"{selected_subfolder}_imp",
        date_prefix = parallel_date,
        output_dir = plots_results_folder_mem,
        max_y = common_max_peak_mem,
        min_y = common_min_peak_mem)

# functional
if mem_files_by_style['functional']:
    graph_pid_mem_peak(
        parallel_csv = mem_files_by_style['functional'][0],
        sequential_csv = func_seq,
        data_label = f"{selected_subfolder}_func",
        date_prefix = parallel_date,
        output_dir = plots_results_folder_mem,
        max_y = common_max_peak_mem,
        min_y = common_min_peak_mem)







