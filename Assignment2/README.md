# Assignment 2: Imperative Vs Functional Paradimg in Java

This project requires a Java version equal or higher than 14

## Running the Experiment

We used build.gradle to automize the running of the experiment. We defined three different types of tasks:
* a run to record memory usage, via shell script - generates a time and a memory report 
* a run to record perf metrics - generates a time and a perf report
* a run to warmup the server 

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
warmupMandelbrot - runAllMandelbrot (on the 18 of may)
warmupSpectralNorm - runAllSpectralNorm (on the 19 of may)

## Plotting scripts