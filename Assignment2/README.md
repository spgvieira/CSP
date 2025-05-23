# Assignment 2: Imperative Vs Functional Paradimg in Java

This project requires a Java version equal or higher than 14

## Running the Experiment

We used build.gradle to automize the running of the experiment. We defined three different types of tasks:
* a run task to record memory usage, via shell script - generates a time and a memory report 
* a run task to record perf metrics - generates a time and a perf report
* a warmup task to get the server started before recording exeperimental values

To run the experiments you can take advantage of the gradlew command, running `./gradlew` and then one of the following defined experiments:
* warmupSepctralNorm - runs warmup task for spectral norm +  records baseline memory 
* warmupMandelbrot - runs warmup task for mandelbrot +  records baseline memory 
* runAllSpectralNorm - runs both memory and perf task on top of Spectral Norm
* runAllMandelbrot - runs both memory and perf task on top of Mandelbrot
* runMemorySpectralNorm - runs memory task on top of Spectral Norm
* runPerfSpectralNorm -  runs perf task on top of Spectral Norm
* runMemoryMandelbrot - runs memory task on top of Mandelbrot
* runPerfMandelbrot - runs perf task on top of Mandelbrot

The numbres shown in the report result from running:
* warmupMandelbrot - runAllMandelbrot (on the 18 of may)
* warmupSpectralNorm - runAllSpectralNorm (on the 19 of may)

## Plotting scripts
## Usage of new_plot_stuff.py

This script generates plots of wall time, perf values and memory usage from experimental data files of type .csv, .txt and .data. It also fixes and adds relevant missing column names for any csv's that may be missing them.

**1. Running the script:**

Execute the script from your terminal using the following command structure:

```bash
python plot_script.py <data_type> <parallel_date> [sequential_date]
```

*   `<data_type>`: Required. Specify the type of data to plot. Use `spec` for spectral norm or `mand` for mandelbrot.
*   `<parallel_date>`: Required. The date (e.g., `19-05`) corresponding to the parallel experiment results you want to plot.
*   `[sequential_date]`: Optional. The date (e.g., `19-05`) corresponding to sequential experiment results to include on the plots. If omitted, only parallel results will be plotted but it is highly recommended to run with results from a sequential date as due to time constraints, there may be some cases in the script where it has not accounted for that sequential results are missing.

**Example:**
In the Assignment2/plots folder, run
```bash
python plot_script.py spec 19-05 19-05
```

This would plot spectral norm data using parallel results from `19-05` and sequential results from `19-05`.

**2. Configuration Settings for new_plot_stuff.py:**

Several dictionaries and variables within the script file control plot appearance and which metrics are plotted. You can modify these directly in the `plot_script.py` file (look for the sections marked like the following: `=== Values for graph customization ===`):

*   `PLOTS_FOLDER_MAP`: Folder mapping for the different output folders for graphs.
*   `thread_colors`: Dictionary mapping thread counts to line colors.
*   `thread_markers`: Dictionary mapping thread counts to plot markers.
*   `thread_markersize`: Dictionary mapping thread counts to marker sizes.
*   `metrics_to_divide_with_input_size`: Set of performance metric names that will be divided by input size squared before plotting.
*   `perf_metrics_collection`: Dictionary controlling which performance metrics are plotted (set value to `True` to enable plotting for a metric, `False` to disable).
*   `mem_metrics_collection`: Dictionary controlling which memory metrics are plotted (`True`/`False`). Note: `pid_mem` is plotted separately as a peak value graph controlled by a different function call near the end of the script.
*   `indep`: Set of performance metric names that will *not* share Y-axis scaling between imperative and functional plots if both are present.
*   `baseline_subtract_true`: Boolean (`True`/`False`) to control whether a defined baseline memory value is subtracted for system memory plots.
*   `baseline_mem_mandelbrot`, `baseline_mem_spec_18_05`, `baseline_mem_spec_19_05`: Define baseline memory values used if `baseline_subtract_true` is `True`.
*   `log_scale`: Boolean (`True`/`False`) to enable or disable log scale for the wall time plot.
*   `log_base`: Integer specifying the base for the log scale if enabled.

Modify these variables directly in the script to customize how the plots look and what is plotted more specifically.
