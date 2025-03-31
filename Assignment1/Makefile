# Variables
EVENTS := cpu-cycles,cache-misses,page-faults,cpu-migrations,dTLB-load-misses,context-switches
REPEAT := 5 #lets run tests on 5 reps!
# THREADS := 1 2 4 8 16 32
THREADS := 32 16 8 4 2 1
# THREADS_CORE_AFF_2 := 1 2 4 8 16
THREADS_CORE_AFF_2 := 16 8 4 2 1
HASHBITS := 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 
PERF := /usr/bin/perf

EXECUTABLES = \
	concurrent independent indep_core_aff_1 indep_core_aff_2 conc_core_aff_1 conc_core_aff_2

# Phony targets
.PHONY: all clean concurrent independent indep_core_aff_1 indep_core_aff_2 conc_core_aff_1 conc_core_aff_2

# Default target
all: $(EXECUTABLES)

# Run with $sudo make clean
# clean:
# 	@echo "Cleaning build artifacts..."
# 	rm -f $(EXECUTABLES)
# 	@echo "Clearing system cache (requires sudo)..."
# 	# sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"
# 	@echo "Clean complete."

# Warm-up

# Independent target
independent:
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

# First independent core affinity target
indep_core_aff_1:
	@echo "Independent core affinity 1"
	@for threads in  $(THREADS); do \
		for hb in $(HASHBITS); do \
			echo "Running: perf stat -e $(EVENTS) --repeat=$(REPEAT) ./build/indep_core_aff_1 $$threads $$hb"; \
			perf stat -e $(EVENTS) -o results/$$(date +"%d_%m")_indep_core_aff_1_results.txt --append --repeat=$(REPEAT) ./build/indep_core_aff_1 $$threads $$hb | tee -a results/$$(date +"%d_%m")_indep_core_aff_1_time.txt & \
			perf_pid=$$!; \
			wait $$perf_pid; \
		done; \
	done
	@echo "Done"

# Second independent core affinity target
indep_core_aff_2:
	@echo "Independent core affinity 2"
	@for threads in  $(THREADS_CORE_AFF_2); do \
		for hb in $(HASHBITS); do \
			echo "Running: perf stat -e $(EVENTS) --repeat=$(REPEAT) ./build/indep_core_aff_2 $$threads $$hb"; \
			perf stat -e $(EVENTS) -o results/$$(date +"%d_%m")_indep_core_aff_2_results.txt --append --repeat=$(REPEAT) ./build/indep_core_aff_2 $$threads $$hb | tee -a results/$$(date +"%d_%m")_indep_core_aff_2_time.txt & \
			perf_pid=$$!; \
			wait $$perf_pid; \
		done; \
	done
  @echo "Done"

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

# First concurrent core affinity target
conc_core_aff_1:
	@echo "Concurrent core affinity 1"
	@for threads in  $(THREADS); do \
		for hb in $(HASHBITS); do \
			echo "Running: perf stat -e $(EVENTS) --repeat=$(REPEAT) ./build/conc_core_aff_1 $$threads $$hb"; \
			perf stat -e $(EVENTS) -o results/$$(date +"%d_%m")_conc_core_aff_1_results.txt --append --repeat=$(REPEAT) ./build/conc_core_aff_1 $$threads $$hb | tee -a results/$$(date +"%d_%m")_conc_core_aff_1_time.txt & \
			perf_pid=$$!; \
			wait $$perf_pid; \
		done; \
	done
	@echo "Done"

# Second independent core affinity target
conc_core_aff_2:
	@echo "Concurrent core affinity 2"
	@for threads in  $(THREADS_CORE_AFF_2); do \
		for hb in $(HASHBITS); do \
			echo "Running: perf stat -e $(EVENTS) --repeat=$(REPEAT) ./build/conc_core_aff_2 $$threads $$hb"; \
			perf stat -e $(EVENTS) -o results/$$(date +"%d_%m")_conc_core_aff_2_results.txt --append --repeat=$(REPEAT) ./build/conc_core_aff_2 $$threads $$hb | tee -a results/$$(date +"%d_%m")_conc_core_aff_2_time.txt & \
			perf_pid=$$!; \
			wait $$perf_pid; \
		done; \
	done
	@echo "Done"