
// Code made by Su Mei Gwen Ho (suho@itu.dk), Sara Vieira (sapi@itu.dk) & Sophus Kaae Merved (some@itu.dk)


#include <iostream>
#include <cmath>
#include <thread>
#include <vector>
#include "tuples.h"
#include <time.h>
#include <chrono>

using Partition = std::vector<std::tuple<int64_t, int64_t>>;
tuple<int64_t, int64_t>* input;
const size_t numTuples = 16777216;

void partitionInput(int start, int end, int numPartitions, vector<Partition>& partitions) {
    for (int i = start; i < end; i++) {
        tuple<int64_t, int64_t> t = input[i];
        int partitionKey = hashFunction(get<0>(t), numPartitions);
        partitions[partitionKey].push_back(t);
    }
}

void cleanup(std::vector<std::vector<Partition>>& threadPartitions) {
    for (auto& threadPartition : threadPartitions) {
        for (auto& partition : threadPartition) {
            partition.clear();
        }
    }
    for(int i=0; i<numTuples; i++) {
        input[i] = make_tuple(0,0);
    }

}

int main(int argc, char* argv[]) {
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

    auto start_clock = std::chrono::steady_clock::now();

    for (int i = 0; i < numThreads; i++) {
        auto start = i * numTuplesPerThread;
        auto end = (i + 1) * numTuplesPerThread;
        std::thread thread(partitionInput, start, end, numPartitions, std::ref(threadPartitions[i]));
        threads.push_back(std::move(thread));
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end_clock = std::chrono::steady_clock::now();
    std::chrono::duration<double> cpu_time_used = end_clock - start_clock;
    printf("%d,%d,%f\n", numThreads, hashBits, cpu_time_used);

    return 0;
}


