#include <iostream>
#include <cmath>
#include <thread>
#include <vector>
#include "tuples.h"
#include <time.h>
#include <atomic>

// TODO: 
// re read code and clean not used
// re evaluate which variables should be public
// missing affinity
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

}

// void cleanup(std::vector<Partition>& partitions) {
//     for (auto& partition : partitions) {
//         partition.clear();
//     }
// }

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

    auto start_clock = std::chrono::steady_clock::now();
    
    for (int i = 0; i < numThreads; i++) {
        auto thread_start = i * numTuplesPerThread;
        auto thread_end = (i + 1) * numTuplesPerThread;
        std::thread thread(partitionInput, thread_start, thread_end, numPartitions, std::ref(partitions), std::ref(partitionIndices));
        threads.push_back(std::move(thread));
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end_clock = std::chrono::steady_clock::now();
    std::chrono::duration<double> cpu_time_used = end_clock - start_clock;
    printf("%d,%d,%f\n", numThreads, hashBits, cpu_time_used);

    // cleanup(partitions);

    return 0;
}
