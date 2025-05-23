# Assignment 1: Data Partitioning and Parallel Processing on Multicore CPUs

This project automates the benchmarking of multiple programs using Linux perf tool to gather performance metrics. It evaluates different threading and hash bit configurations under various scheduling and core affinity settings.

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
