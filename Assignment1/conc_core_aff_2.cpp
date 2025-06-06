//this core affinity script takes multiple integer arguments to allow it to compute the cpu nr needed
//run using eg $./affinitynumaexample 0 1 2 3 for 4 threads
//pins each thread to the consecutive core in the same NUMA node (we are only using even nr cores in NUMA node 0)
//comment out print statments when running the experiment!

// Code made by Su Mei Gwen Ho (suho@itu.dk), Sara Vieira (sapi@itu.dk) & Sophus Kaae Merved (some@itu.dk)

#include <iostream>
#include <cmath>
#include <thread>
#include <vector>
#include "tuples.h"
#include <time.h>
#include <atomic>
#include <pthread.h>

using Partition = std::vector<std::tuple<int64_t, int64_t>>;
tuple<int64_t, int64_t>* input;

void partitionInput(int start, int end, int numPartitions, 
    vector<Partition>& partitions, std::vector<std::atomic<size_t>>& locks) {
    for (int i = start; i < end; i++) {
        tuple<int64_t, int64_t> t = input[i];
        int partitionKey = hashFunction(get<0>(t), numPartitions);
        size_t index = locks[partitionKey].fetch_add(1, memory_order_relaxed);
        partitions[partitionKey][index] = t;
    }
    // std::cout << "Thread #" << numThread << ": on CPU " 
    //             << sched_getcpu() << "\n"; 
}

int main(int argc, char* argv[]) {
    const size_t numTuples = 16777216;
    input = makeInput(numTuples);

    const int numThreads = atoi(argv[1]);
    const int numTuplesPerThread = numTuples / numThreads;
    const int hashBits = atoi(argv[2]);
    const int numPartitions = 1 << hashBits;
    const int sizePartition = numTuples/numPartitions * 1.5;

    std::vector<std::thread> threads;
    std::vector<Partition> partitions(numPartitions);

    // Pre-allocate memory to prevent dynamic resizing overhead
    for (auto& partition : partitions)
        partition.reserve(sizePartition);  

    std::vector<std::atomic<size_t>> partitionIndices(numPartitions);

    cpu_set_t cpuset[numThreads];
    for(int i=0; i < numThreads; i++) {
        CPU_ZERO(&cpuset[i]);
        CPU_SET(i*2, &cpuset[i]); //adds the CPU core with index i*2 to the cpuset
    }

    auto start_clock = std::chrono::steady_clock::now();
    
    for (int i = 0; i < numThreads; i++) {
        auto thread_start = i * numTuplesPerThread;
        auto thread_end = (i + 1) * numTuplesPerThread;
        std::thread thread(partitionInput, thread_start, thread_end, numPartitions, std::ref(partitions), std::ref(partitionIndices));
        threads.push_back(std::move(thread));
        //this pins Thread #0: on CPU 0, Thread #1: on CPU 2, Thread #2: on CPU 4, etc
        int rc = pthread_setaffinity_np(threads[i].native_handle(),
                                        sizeof(cpu_set_t), &cpuset[i]);
        if (rc != 0) {
            cerr << "Error calling pthread_setaffinity_np: " << rc << endl;
        }
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end_clock = std::chrono::steady_clock::now();
    std::chrono::duration<double> cpu_time_used = end_clock - start_clock;
    printf("%d,%d,%f\n", numThreads, hashBits, cpu_time_used);

    return 0;
}
