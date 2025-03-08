#include <iostream>
#include <cmath>
#include <thread>
#include <vector>
#include "tuples.h"
#include <time.h>

// TODO: 
// re read code and clean not used
// re evaluate which variables should be public
// missing affinity setting

tuple<int64_t, int64_t>* input;

void partitionInput(int numThread, int start, int end, int numPartitions, 
    vector<vector<tuple<int64_t, int64_t>>>& partitions) {
    for (int i = start; i < end; i++) {
        tuple<int64_t, int64_t> t = input[i];
        int partitionKey = hashFunction(get<0>(t), numPartitions);
        partitions[partitionKey].push_back(t);
    }
}

void cleanup(std::vector<std::vector<std::vector<std::tuple<int64_t, int64_t>>>>& threadPartitions) {
    for (auto& threadPartition : threadPartitions) {
        for (auto& partition : threadPartition) {
            partition.clear();
        }
    }
}

int main(int argc, char* argv[]) {
    const size_t numTuples = 16777216;
    input = makeInput(numTuples);

    const int numThreads = atoi(argv[1]);
    const int numTuplesPerThread = numTuples / numThreads;

    const int hashBits = atoi(argv[2]);
    const int numPartitions = 1 << hashBits;
    const int sizePartition = numTuples/numPartitions * 1.5;

    std::vector<std::thread> threads(numThreads);
    std::vector<std::vector<std::vector<std::tuple<int64_t, int64_t>>>> threadPartitions(numThreads, 
        std::vector<std::vector<std::tuple<int64_t, int64_t>>>(numPartitions));

    // Pre-allocate memory to prevent dynamic resizing overhead
    for (auto& threadPartition : threadPartitions)
        for (auto& partition : threadPartition)
            partition.reserve(sizePartition);  

    clock_t start_clock, end_clock;
    double cpu_time_used;


    start_clock = clock();

    for (int i = 0; i < numThreads; i++) {
        auto start = i * numTuplesPerThread;
        auto end = (i + 1) * numTuplesPerThread;

        threads[i] = std::thread(partitionInput, i, start, end, numPartitions, std::ref(threadPartitions[i]));
    }

    for (auto& t : threads) {
        t.join();
    }

    end_clock = clock();
    cpu_time_used = ((double)(end_clock - start_clock)) / CLOCKS_PER_SEC;
    printf("no threads %d no hash bits %d time:%f\n", numThreads, hashBits, cpu_time_used);

    cleanup(threadPartitions);

    return 0;
}
