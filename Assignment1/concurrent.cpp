#include <iostream>
#include <cmath>
#include <thread>
#include <mutex>
#include <vector>
#include "tuples.h"
#include <time.h>
#include <atomic>

// TODO: 
// re read code and clean not used
// re evaluate which variables should be public
// missing affinity

tuple<int64_t, int64_t>* input;

void partitionInput(int numThread, int start, int end, int numPartitions, 
    vector<vector<tuple<int64_t, int64_t>>>& partitions, std::vector<std::atomic<size_t>>& locks) {

    for (int i = start; i < end; i++) {
        tuple<int64_t, int64_t> t = input[i];
        int partitionKey = hashFunction(get<0>(t), numPartitions);
        // locks[partitionKey].lock();
        size_t index = locks[partitionKey].fetch_add(1, memory_order_relaxed);
        partitions[partitionKey][index] = t;
        // partitions[partitionKey].push_back(t);
        // locks[partitionKey].unlock();
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
    std::vector<std::vector<std::tuple<int64_t, int64_t>>> partitions(numPartitions);
    // std::vector<std::mutex> locks(numPartitions);

    // Pre-allocate memory to prevent dynamic resizing overhead
    for (auto& partition : partitions)
        partition.reserve(sizePartition);  

    std::vector<std::atomic<size_t>> partitionIndices(numPartitions);

    auto start_clock = std::chrono::steady_clock::now();
    
    for (int i = 0; i < numThreads; i++) {
        auto thread_start = i * numTuplesPerThread;
        auto thread_end = (i + 1) * numTuplesPerThread;

        threads[i] = std::thread(partitionInput, i, thread_start, thread_end, numPartitions, std::ref(partitions), std::ref(partitionIndices));
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end_clock = std::chrono::steady_clock::now();
    std::chrono::duration<double> cpu_time_used = end_clock - start_clock;
    printf("%d; %d; %f\n", numThreads, hashBits, cpu_time_used);

    return 0;
}
