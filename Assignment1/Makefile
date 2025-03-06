# Variables
RESULTS_FILE := result.txt
EVENTS := cpu-cycles,cache-misses,page-faults,cpu-migrations,dTLB-load-misses,context-switches
REPEAT := 2
THREADS := 1 2 4 8 16 32
HASHBITS := 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
PERF := /usr/bin/perf

# Phony targets
.PHONY: all clean concurrent independent 

# Default target
all: concurrent independent

# Clean target
clean:
	@echo "Cleaning up..."
	@rm -f $(RESULTS_FILE)

# Concurrent target
concurrent:
	@echo "Concurrent"
	@for threads in $(THREADS); do \
		for hb in $(HASHBITS); do \
			echo "Running: perf stat -e $(EVENTS) --repeat=$(REPEAT) ./build/concurrent $$threads $$hb"; \
			$(PERF) stat -e $(EVENTS) -o concurrent_results.txt --append --repeat=$(REPEAT) ./build/concurrent $$threads $$hb | tee -a concurrent_time.txt & \
			perf_pid=$$!; \
			wait $$perf_pid; \
		done; \
	done
	@echo "Done"

# Independent target
independent:
	@echo "Independent"
	@for threads in  $(THREADS); do \
		for hb in $(HASHBITS); do \
			echo "Running: perf stat -e $(EVENTS) --repeat=$(REPEAT) ./build/independent $$threads $$hb"; \
			perf stat -e $(EVENTS) -o independent_results.txt --append --repeat=$(REPEAT) ./build/independent $$threads $$hb | tee -a independent_time.txt & \
			perf_pid=$$!; \
			wait $$perf_pid; \
		done; \
	done
	@echo "Done"