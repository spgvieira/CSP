//this core affinity script takes multiple integer arguments to allow it to compute the cpu nr needed
//run using eg $./affinitynumaexample 0 1 2 3 for 4 threads
//pins each thread to the consecutive core in the same NUMA node
//comment out print statments when running the experiment!


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

void partitionInput(int start, int end, int numPartitions, vector<Partition>& partitions) {
    for (int i = start; i < end; i++) {
        tuple<int64_t, int64_t> t = input[i];
        int partitionKey = hashFunction(get<0>(t), numPartitions);
        partitions[partitionKey].push_back(t);
        std::cout << "Thread #" << i << ": on CPU " 
                << sched_getcpu() << "\n"; 
    }
}

void cleanup(std::vector<std::vector<Partition>>& threadPartitions) {
    for (auto& threadPartition : threadPartitions) {
        for (auto& partition : threadPartition) {
            partition.clear();
        }
    }
}

int main(int argc, char* argv[]) {
    //needs nr threads, nr hashbits, n at least 1 cpu id
    // if (argc < 3) {
    //     cout << "Require atleast 3 arguments" << endl;
    //     return -1;
    // }
    const size_t numTuples = 16777216;
    input = makeInput(numTuples);

    const int numThreads = atoi(argv[1]);
    const int numTuplesPerThread = numTuples / numThreads;
    const int hashBits = atoi(argv[2]);
    const int numPartitions = 1 << hashBits;
    const int sizePartition = numTuplesPerThread/numPartitions * 1.5;

    //this check is here if needed
    // if (argc < numThreads + 2){
    //     cout << "CPU id must be specified for every thread" << endl;
    //     return -1;
    // }
    
    std::vector<std::thread> threads;
    std::vector<std::vector<Partition>> threadPartitions(numThreads, 
        std::vector<Partition>(numPartitions));

    // Pre-allocate memory to prevent dynacdmic resizing overhead
    for (auto& threadPartition : threadPartitions)
        for (auto& partition : threadPartition)
            partition.reserve(sizePartition);  

    auto start_clock = std::chrono::steady_clock::now();

    for (int i = 0; i < numThreads; i++) {
        auto start = i * numTuplesPerThread;
        auto end = (i + 1) * numTuplesPerThread;
        std::thread thread(partitionInput, start, end, numPartitions, std::ref(threadPartitions[i]));
        threads.push_back(std::move(thread));

        // Create a cpu_set_t object representing a set of CPUs.
        // Clear it and mark only CPU i as set.
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        //loop thru the args given to compute cpu id
        CPU_SET(atoi(argv[3 + i]), &cpuset);
        int rc = pthread_setaffinity_np(threads[i].native_handle(),
                                        sizeof(cpu_set_t), &cpuset);
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

    cleanup(threadPartitions);

    return 0;
}


