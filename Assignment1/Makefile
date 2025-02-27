# Variables
PATH := result.txt
# EVENTS := cpu-cycles,cache-misses,page-faults,cpu-migrations,L1-dcache-load-misses,dTLB-load-misses,LLC-load-misses
EVENTS := cpu-cycles,cache-misses,page-faults,cpu-migrations,L1-dcache-load-misses,dTLB-load-misses,LLC-load-misses,power/energy-pkg/,power/energy-ram/
REPEAT := 5
THREADS := 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
HASHBITS := 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
PERF := /bin/perf


# Phony targets
.PHONY: all clean concurrent independent 

# Default target
all: concurrent independent test

# Clean target
clean:
	@echo "Cleaning up..."
	@rm -f $(PATH)

# Concurrent target
concurrent:
	@echo "Concurrent"
	@for threads in $(THREADS); do \
		for hb in $(HASHBITS); do \
			echo "Running: perf stat -e $(EVENTS) -o $(PATH) --append --repeat=$(REPEAT) ./build/concurrent $$threads $$hb"; \
			$(PERF) stat -e $(EVENTS) -o $(PATH) --append --repeat=$(REPEAT) ./build/concurrent $$threads $$hb & \
			perf_pid=$$!; \
			wait $$perf_pid; \
		done; \
	done
	@echo "Done"

# Independent target
independent:
	@echo "Independent"
	@for threads in $(shell seq 1 32); do \
		for hb in $(shell seq 1 18); do \
			perf stat -e $(EVENTS) -o $(PATH) --append --repeat=$(REPEAT) ./build/independent $$threads $$hb & \
			perf_pid=$$!; \
			wait $$perf_pid; \
		done; \
	done