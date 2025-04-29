# --- Load Data ---
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
csv_filename = '28_04_imp_parallel_perf.csv'  # Make sure your CSV file is named this

# Define thread colors and markers
# Extended to cover up to 128 threads
thread_colors = {
    1: 'cyan', 2: 'green', 4: 'magenta', 8: 'orange',
    16: 'blue', 32: 'red', 64: 'purple', 128: 'brown'
}

thread_markers = {
    1: 'o', 2: 's', 4: '^', 8: 'D',
    16: 'X', 32: '*', 64: 'p', 128: 'h' # p=pentagon, h=hexagon
}

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

# --- Prepare for Categorical-like X-axis ---
# 1. Get unique sorted InputSize values
unique_input_sizes = sorted(plot_data['InputSize'].unique())
# 2. Create mapping from InputSize to index (0, 1, 2...)
input_size_to_index = {size: i for i, size in enumerate(unique_input_sizes)}
# 3. Create the list of indices for tick positions
x_tick_indices = range(len(unique_input_sizes))

# --- Plotting ---

# 1. Instructions Per Cycle (IPC) Plot
plt.figure(figsize=(12, 7)) # Slightly wider figure for potentially many labels
for threads in thread_counts:
    subset = plot_data[plot_data['Threads'] == threads]
    # Sort by InputSize to ensure lines are drawn correctly
    subset = subset.sort_values('InputSize')
    # Map the actual InputSize values to their corresponding indices for plotting
    x_plot_values = subset['InputSize'].map(input_size_to_index)

    # Get color and marker, provide defaults if thread count not in dict
    color = thread_colors.get(threads, 'black') # Default to black
    marker = thread_markers.get(threads, '.')   # Default to small dot

    plt.plot(x_plot_values, subset['ipc'],
             marker=marker,
             markersize=8, # You can adjust marker size here
             linestyle='-',
             color=color,
             label=f'{threads} Threads')

plt.xlabel('Input Size')
plt.ylabel('Instructions Per Cycle (IPC)')
plt.title('IPC vs. Input Size by Thread Count')
# Set the ticks at the index positions, but label them with original InputSize values
plt.xticks(ticks=x_tick_indices, labels=unique_input_sizes)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Remove x-axis margins: start at index 0, end at the last index
plt.xlim(left=0, right=len(unique_input_sizes) - 1)


# 2. Page Faults Plot
plt.figure(figsize=(12, 7)) # Slightly wider figure
for threads in thread_counts:
    subset = plot_data[plot_data['Threads'] == threads]
    # Sort by InputSize
    subset = subset.sort_values('InputSize')
    # Map the actual InputSize values to their corresponding indices for plotting
    x_plot_values = subset['InputSize'].map(input_size_to_index)

    # Get color and marker, provide defaults if thread count not in dict
    color = thread_colors.get(threads, 'black') # Default to black
    marker = thread_markers.get(threads, '.')   # Default to small dot

    plt.plot(x_plot_values, subset['page_faults'],
             marker=marker,
             markersize=8, # You can adjust marker size here
             linestyle='-',
             color=color,
             label=f'{threads} Threads')

plt.xlabel('Input Size')
plt.ylabel('Average Page Faults')
plt.title('Page Faults vs. Input Size by Thread Count')
# Set the ticks at the index positions, but label them with original InputSize values
plt.xticks(ticks=x_tick_indices, labels=unique_input_sizes)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Remove x-axis margins: start at index 0, end at the last index
plt.xlim(left=0, right=len(unique_input_sizes) - 1)


# --- Show Plots ---
plt.tight_layout() # Adjust layout to prevent labels overlapping
plt.show()

print("Processing complete. Plots should be displayed.")