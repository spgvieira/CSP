import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import re #for naming output files
from datetime import datetime #for naming output files
import matplotlib.ticker as ticker #for more stylish/even numbered tickers along y-axis
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
    "dtlb_load_misses",
    "page_faults"
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
log_scale = True
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

def compute_max_y_norm(parallel_csvs, sequential_csvs=None, metrics=None, metrics_to_normalize=None):
    """
    Determine maximum grouped mean across parallel and sequential CSV files.
    Can optionally normalize metrics by input size squared BEFORE finding max.

    parallel_csvs: list of file paths
    sequential_csvs: list of file paths (optional)
    metrics: list of column names to consider
    metrics_to_normalize: set or list of metric names to normalize by input**2
                          BEFORE finding the maximum.
    Returns: dict of {metric: max_value} or single float if len(metrics)==1
    """
    if metrics is None:
        metrics = ['avg_time']
    if metrics_to_normalize is None:
        metrics_to_normalize = set() # Use a set for efficient lookup
    else:
         metrics_to_normalize = set(metrics_to_normalize) # Convert to set

    max_vals = {m: 0 for m in metrics}

    # Helper to update a metric’s max
    def _update_max(m, value):
        if value is not None and not np.isnan(value):
             max_vals[m] = max(max_vals[m], value)


    # Function to process a single DataFrame (either parallel or sequential)
    def process_df(df, is_parallel):
        if df.empty:
            return # Nothing to process

        # Ensure 'input' column exists if any metric needs normalization
        needs_input_for_norm = any(m in metrics_to_normalize for m in metrics)
        if needs_input_for_norm and 'input' not in df.columns:
             print(f"Warning: 'input' column missing in file processed by compute_max_y. Cannot normalize metrics.")
             metrics_to_normalize.clear() # Disable normalization if input is missing

        # Calculate grouped means for each metric
        for m in metrics:
            if m not in df.columns:
                continue # Metric not in this file

            if m in metrics_to_normalize and 'input' in df.columns:
                # --- Handle normalization before grouping ---
                # Create a temporary normalized series
                normalized_series = pd.Series(np.nan, index=df.index)
                # Avoid division by zero
                non_zero_input_mask = df['input'] != 0
                if non_zero_input_mask.any():
                    normalized_series.loc[non_zero_input_mask] = df.loc[non_zero_input_mask, m] / (df.loc[non_zero_input_mask, 'input'] ** 2)

                # Group the temporary normalized series
                if is_parallel and 'thread' in df.columns:
                    # For parallel, group by input and thread
                    temp_df = df[['input', 'thread']].copy()
                    temp_df['normalized_metric'] = normalized_series
                    grp = temp_df.groupby(['input', 'thread'])['normalized_metric'].mean()
                else: # Sequential or parallel without thread (shouldn't happen for parallel data)
                    # For sequential, group only by input
                    temp_df = df[['input']].copy()
                    temp_df['normalized_metric'] = normalized_series
                    grp = temp_df.groupby('input')['normalized_metric'].mean()

                # Update max value using the mean of the normalized values
                if not grp.empty:
                     _update_max(m, grp.max())

            else:
                # --- Handle metrics that are NOT normalized ---
                if is_parallel and 'thread' in df.columns:
                     grp = df.groupby(['input', 'thread'])[m].mean()
                else: # Sequential
                     grp = df.groupby('input')[m].mean()

                if not grp.empty:
                     _update_max(m, grp.max())


    # Process parallel CSVs
    for csv in parallel_csvs:
        try:
            df = pd.read_csv(csv)
            process_df(df, is_parallel=True)
        except Exception as e:
            print(f"Error reading or processing {csv} in compute_max_y: {e}")
            continue


    # Process sequential CSVs
    if sequential_csvs:
        for csv in sequential_csvs:
            try:
                df = pd.read_csv(csv)
                # Sequential files typically don't have a 'thread' column,
                # so pass is_parallel=False
                process_df(df, is_parallel=False)
            except Exception as e:
                 print(f"Error reading or processing {csv} in compute_max_y: {e}")
                 continue


    # Filter out metrics for which no non-NaN max was found (e.g. column missing in all files)
    final_max_vals = {m: v for m, v in max_vals.items() if v != 0 or all(m not in pd.read_csv(f).columns for f in parallel_csvs + (sequential_csvs or []))}

    # If metrics had initial max_val 0 and weren't present in any file, keep them at 0.
    for m in metrics:
        if m not in final_max_vals and m in max_vals:
             final_max_vals[m] = max_vals[m]


    return final_max_vals if len(metrics) > 1 else final_max_vals[metrics[0]]

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
    ax.set_ylabel('Wall Time Log Scale (ms)')
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


def graph_perf_values(parallel_csv, sequential_csv=None, data_label="", date_prefix="", metrics_dict=None, max_y=None,
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
        ax.set_ylabel(metric.replace("_", " ").title())
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
        parts = [date_prefix, data_label, metric + ".png"] if date_prefix else [data_label, metric + ".png"]
        filename = "_".join(p for p in parts if p)
        outdir = output_dir or plots_results_folder_time
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, filename)
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"saved perf metric plot '{metric}' → {outpath}")

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
            df_seq['free_mem'] = - baseline_mem - df_seq['free_mem']
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

        ylabel_text = metric.replace("_", " ").title()
        if metric == "free_mem" and baseline_subtract_true:
             ylabel_text = "Memory Used (Baseline - Free Mem)" # Or similar descriptive label
        ylabel_text += " (MB)" # Add units
        ax.set_ylabel(ylabel_text)
        ax.set_xticks(list(x_index.values()))
        ax.set_xticklabels(input_sizes)
        ax.set_xlim(0, len(input_sizes) - 1)
        ax.set_ylim(y0, y1)
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
    max_y_perf = compute_max_y_norm(combined_par, combined_seq, metrics=metrics_list, metrics_to_normalize = metrics_to_divide_with_input_size)
else:
    # fallback to per‐style: here we only need two separate dicts
    max_y_imp  = compute_max_y_norm(imp_par_list,  imp_seq_list,  metrics=metrics_list, metrics_to_normalize = metrics_to_divide_with_input_size) if imp_par_list else {}
    max_y_func = compute_max_y_norm(func_par_list, func_seq_list, metrics=metrics_list, metrics_to_normalize = metrics_to_divide_with_input_size) if func_par_list else {}
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
        metrics_to_normalize_by_input = metrics_to_divide_with_input_size
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
        metrics_to_normalize_by_input = metrics_to_divide_with_input_size
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


# Compute the max_y time value across different csv's
'''def compute_max_y(parallel_csvs, sequential_csvs=None):
    """
    Determine the maximum average time across multiple parallel and sequential CSV files.
    parallel_csvs: list of file paths for parallel runs
    sequential_csvs: list of file paths for sequential runs (optional)
    Returns a float max value.
    """
    max_vals = []
    for csv in parallel_csvs:
        try:
            grp = calculate_averages_parallel(csv)
            max_vals.append(grp.max())
        except Exception:
            continue
    if sequential_csvs:
        for csv in sequential_csvs:
            try:
                seq = calculate_averages_seq(csv)
                max_vals.append(seq.max())
            except Exception:
                continue
    return max(max_vals) if max_vals else 0'''

# === Debug Output ===
'''print("\n--- Parallel Files ---")
print("Time:", time_files)
print("Perf:", perf_files)
print("Mem:", mem_files)
print("Mem (PID):", mem_pid_files)

print("\n> Functional")
print("Time:", time_files_by_style['functional'])
print("Perf:", perf_files_by_style['functional'])
print("Mem:", mem_files_by_style['functional'])
print("Mem (PID):", mem_pid_files_by_style['functional'])

print("\n> Imperative")
print("Time:", time_files_by_style['imperative'])
print("Perf:", perf_files_by_style['imperative'])
print("Mem:", mem_files_by_style['imperative'])
print("Mem (PID):", mem_pid_files_by_style['imperative'])

if sequential_date:
    print("\n--- Sequential Files ---")
    print("Time:", time_files_seq)
    print("Perf:", perf_files_seq)
    print("Mem:", mem_files_seq)
    print("Mem (PID):", mem_pid_files_seq)

    print("\n> Functional")
    print("Time:", time_files_seq_by_style['functional'])
    print("Perf:", perf_files_seq_by_style['functional'])
    print("Mem:", mem_files_seq_by_style['functional'])
    print("Mem (PID):", mem_pid_files_seq_by_style['functional'])

    print("\n> Imperative")
    print("Time:", time_files_seq_by_style['imperative'])
    print("Perf:", perf_files_seq_by_style['imperative'])
    print("Mem:", mem_files_seq_by_style['imperative'])
    print("Mem (PID):", mem_pid_files_seq_by_style['imperative'])'''






