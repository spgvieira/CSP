CXX = gcc
CXXFLAGS = -g

all: caching vertical horizontal unrolling original_loop

caching: caching.c
	$(CXX) $(CXXFLAGS) -O0 -o build/caching-0 caching.c
	$(CXX) $(CXXFLAGS) -O3 -o build/caching-3 caching.c

vertical: vertical.c
	$(CXX) $(CXXFLAGS) -O0 -o build/vertical-0 vertical.c
	$(CXX) $(CXXFLAGS) -O3 -o build/vertical-3 vertical.c

horizontal: horizontal.c
	$(CXX) $(CXXFLAGS) -O0 -o build/horizontal-0 horizontal.c
	$(CXX) $(CXXFLAGS) -O3 -o build/horizontal-3 horizontal.c

unrolling: unrolled.c
	$(CXX) $(CXXFLAGS) -O0 -o build/unrolled-0 unrolled.c
	$(CXX) $(CXXFLAGS) -O3 -o build/unrolled-3 unrolled.c

original_loop: original_loop.c
	$(CXX) $(CXXFLAGS) -O0 -o build/original_loop-0 original_loop.c
	$(CXX) $(CXXFLAGS) -O3 -o build/original_loop-3 original_loop.c

.PHONY: all caching vertical horizontal unrolling original_loop
