# Variables
RESULTS_FILE := result.txt
EVENTS := cpu-cycles,cache-misses,page-faults,cpu-migrations,dTLB-load-misses,context-switches
REPEAT := 2
THREADS := 1 2 4 8 16 32
THREADS_CORE_AFF := 1 2 4 8 16
HASHBITS := 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 
PERF := /usr/bin/perf

EXECUTABLES = \
	concurrent independent indep_core_aff_1 indep_core_aff_2

# Phony targets
.PHONY: all clean concurrent independent #indep_core_aff_1 indep_core_aff_2

# Default target
all: $(EXECUTABLES)

# Clean target
clean:
	@echo "Cleaning up..."
	@rm -f $(RESULTS_FILE)

# Warm-up

# Concurrent target
concurrent:
	@echo "Concurrent"
	@for threads in $(THREADS); do \
		for hb in $(HASHBITS); do \
			echo "Running: perf stat -e $(EVENTS) --repeat=$(REPEAT) ./build/concurrent $$threads $$hb"; \
			$(PERF) stat -e $(EVENTS) -o results/$$(date +"%d_%m")_concurrent_results.txt --append --repeat=$(REPEAT) ./build/concurrent $$threads $$hb | tee -a results/$$(date +"%d_%m")_concurrent_time.txt & \
			perf_pid=$$!; \
			wait $$perf_pid; \
		done; \
	done
	@echo "Done"

# Independent target
independent:
# indep_core_aff_1:
	@echo "Independent"
	@for threads in  $(THREADS); do \
		for hb in $(HASHBITS); do \
			echo "Running: perf stat -e $(EVENTS) --repeat=$(REPEAT) ./build/independent $$threads $$hb"; \
			perf stat -e $(EVENTS) -o results/$$(date +"%d_%m")_independent_results.txt --append --repeat=$(REPEAT) ./build/independent $$threads $$hb | tee -a results/$$(date +"%d_%m")_independent_time.txt & \
			perf_pid=$$!; \
			wait $$perf_pid; \
		done; \
	done
	@echo "Done"

# First core affinity target
indep_core_aff_1:
	@echo "Independent core affinity 1"
	@for threads in  $(THREADS); do \
		for hb in $(HASHBITS); do \
			echo "Running: perf stat -e $(EVENTS) --repeat=$(REPEAT) ./build/indep_core_aff_1 $$threads $$hb"; \
			perf stat -e $(EVENTS) -o results/$$(date +"%d_%m")_indep_core_aff_1.txt --append --repeat=$(REPEAT) ./build/indep_core_aff_1 $$threads $$hb | tee -a results/$$(date +"%d_%m")_indep_core_aff_1_time.txt & \
			perf_pid=$$!; \
			wait $$perf_pid; \
		done; \
	done
	@echo "Done"

# Second core affinity target
indep_core_aff_2:
	@echo "Independent core affinity 2"
	@for threads in  $(THREADS_CORE_AFF); do \
		for hb in $(HASHBITS); do \
			echo "Running: perf stat -e $(EVENTS) --repeat=$(REPEAT) ./build/indep_core_aff_2 $$threads $$hb"; \
			perf stat -e $(EVENTS) -o results/$$(date +"%d_%m")_indep_core_aff_2.txt --append --repeat=$(REPEAT) ./build/indep_core_aff_2 $$threads $$hb | tee -a results/$$(date +"%d_%m")_indep_core_aff_2_time.txt & \
			perf_pid=$$!; \
			wait $$perf_pid; \
		done; \
	done
	@echo "Done"