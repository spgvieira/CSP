//tomorrow: clean up end of code - comment out merging ?
//check time being taken - wall time or clock time?
//do a quick run, check if results are better?

//add core affinity



#include <iostream>
#include <cmath>
#include <thread>
#include <vector>
#include "tuples.h"
#include <time.h>
#include <cstring> // for memcpy
#include <chrono>

using Partition = std::vector<std::tuple<int64_t, int64_t>>;
tuple<int64_t, int64_t>* input;

void partitionInput(int numThread, int start, int end, int numPartitions, vector<Partition>& partitions) {
    for (int i = start; i < end; i++) {
        tuple<int64_t, int64_t> t = input[i];
        int partitionKey = hashFunction(get<0>(t), numPartitions);
        partitions[partitionKey].push_back(t);
    }
}

int main(int argc, char* argv[]) {
    const size_t numTuples = 16777216;
    input = makeInput(numTuples);

    const int numThreads = atoi(argv[1]);
    const int numTuplesPerThread = numTuples / numThreads;

    const int hashBits = atoi(argv[2]);
    const int numPartitions = 1 << hashBits;
    const int sizePartition = numTuples / numPartitions * 1.5;

    std::vector<std::thread> threads;
    std::vector<std::vector<Partition>> threadPartitions(numThreads);

    // Resizing outside the loop:
    for (int i = 0; i < numThreads; i++) {
        threadPartitions[i].resize(numPartitions);
        for (auto &part : threadPartitions[i]){
            part.reserve(sizePartition);
        }
    }

    //clock_t start_clock, end_clock;
    //double cpu_time_used;

    //start_clock = clock();
    auto start_clock = std::chrono::steady_clock::now();

    for (int i = 0; i < numThreads; i++) {
        auto start = i * numTuplesPerThread;
        auto end = (i + 1) * numTuplesPerThread;
        threads.emplace_back(partitionInput, i, start, end, numPartitions, std::ref(threadPartitions[i]));
    }

    for (auto& t : threads) {
        t.join();
    }
    auto end_clock = std::chrono::steady_clock::now();
    std::chrono::duration<double> cpu_time_used = end_clock - start_clock;
    printf("%d,%d,%f\n", numThreads, hashBits, cpu_time_used);

    // Merge thread-local partitions into final partitions.
    // std::vector<Partition> finalPartitions(numPartitions);
    // for (auto& partition : finalPartitions) {
    //     partition.reserve(numTuples / numPartitions);
    // }

    // for (int threadID = 0; threadID < numThreads; threadID++) {
    //     for (int partitionID = 0; partitionID < numPartitions; partitionID++) {
    //         finalPartitions[partitionID].insert(finalPartitions[partitionID].end(), threadPartitions[threadID][partitionID].begin(), threadPartitions[threadID][partitionID].end());
    //     }
    // }

    //end_clock = clock();
    //cpu_time_used = ((double)(end_clock - start_clock)) / CLOCKS_PER_SEC;
    //printf("no threads %d no hash bits %d time:%f\n", numThreads, hashBits, cpu_time_used);

    return 0;
}