//this core affinity script pins every thread to the corresponding core nr 
//comment out print statments when running the experiment!

// Code made by Su Mei Gwen Ho (suho@itu.dk), Sara Vieira (sapi@itu.dk) & Sophus Kaae Merved (some@itu.dk)

#include <iostream>
#include <cmath>
#include <thread>
#include <vector>
#include "tuples.h"
#include <time.h>
#include <chrono>
#include <pthread.h>

using Partition = std::vector<std::tuple<int64_t, int64_t>>;
tuple<int64_t, int64_t>* input;

void partitionInput(int numThread, int start, int end, int numPartitions, vector<Partition>& partitions) {
    for (int i = start; i < end; i++) {
        tuple<int64_t, int64_t> t = input[i];
        int partitionKey = hashFunction(get<0>(t), numPartitions);
        partitions[partitionKey].push_back(t);
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
    const int sizePartition = numTuplesPerThread/numPartitions * 1.5;

    std::vector<std::thread> threads;
    std::vector<std::vector<Partition>> threadPartitions(numThreads, 
        std::vector<Partition>(numPartitions));

    // Pre-allocate memory to prevent dynacdmic resizing overhead
    for (auto& threadPartition : threadPartitions)
        for (auto& partition : threadPartition)
            partition.reserve(sizePartition);  

    cpu_set_t cpuset[numThreads]; //represents a set of CPUs
    for(int i=0; i < numThreads; i++) {
        CPU_ZERO(&cpuset[i]);
        CPU_SET(i, &cpuset[i]); //adds the CPU core with index i to the cpuset
    }

    auto start_clock = std::chrono::steady_clock::now();

    for (int i = 0; i < numThreads; i++) {
        auto start = i * numTuplesPerThread;
        auto end = (i + 1) * numTuplesPerThread;
        std::thread thread(partitionInput, i, start, end, numPartitions, std::ref(threadPartitions[i]));
        threads.push_back(std::move(thread));
        //this pins Thread #0: on CPU 0, Thread #1: on CPU 1
        int rc = pthread_setaffinity_np(threads[i].native_handle(),
                                        sizeof(cpu_set_t), &cpuset[i]);
        if (rc != 0) {
        std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
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


