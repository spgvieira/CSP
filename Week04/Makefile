CXX = gcc
CXXFLAGS = -g

all: caching vertical horizontal unrolling original_loop

caching: caching.c
	$(CXX) $(CXXFLAGS) -O0 -o caching-0 caching.c
	$(CXX) $(CXXFLAGS) -O3 -o caching-3 caching.c

vertical: vertical.c
	$(CXX) $(CXXFLAGS) -O0 -o vertical-0 vertical.c
	$(CXX) $(CXXFLAGS) -O3 -o vertical-3 vertical.c

horizontal: horizontal.c
	$(CXX) $(CXXFLAGS) -O0 -o horizontal-0 horizontal.c
	$(CXX) $(CXXFLAGS) -O3 -o horizontal-3 horizontal.c

unrolling: unrolled.c
	$(CXX) $(CXXFLAGS) -O0 -o unrolled-0 unrolled.c
	$(CXX) $(CXXFLAGS) -O3 -o unrolled-3 unrolled.c

original_loop: original_loop.c
	$(CXX) $(CXXFLAGS) -O0 -o original_loop-0 original_loop.c
	$(CXX) $(CXXFLAGS) -O3 -o original_loop-3 original_loop.c

.PHONY: all caching vertical horizontal unrolling original_loop
