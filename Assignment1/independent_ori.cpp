#include <iostream>
#include <cmath>
#include <thread>
#include <vector>
#include "tuples.h"
#include <time.h>
#include <chrono>

// TODO: 
// re read code and clean not used
// re evaluate which variables should be public
// missing affinity setting

using Partition = std::vector<std::tuple<int64_t, int64_t>>;
tuple<int64_t, int64_t>* input;

void partitionInput(int numThread, int start, int end, int numPartitions, vector<Partition>& partitions) {
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
    std::vector<std::vector<Partition>> threadPartitions(numThreads, 
        std::vector<Partition>(numPartitions));

    // Pre-allocate memory to prevent dynamic resizing overhead
    for (auto& threadPartition : threadPartitions)
        for (auto& partition : threadPartition)
            partition.reserve(sizePartition);  

    auto start_clock = std::chrono::steady_clock::now();

    for (int i = 0; i < numThreads; i++) {
        auto start = i * numTuplesPerThread;
        auto end = (i + 1) * numTuplesPerThread;
        std::thread thread(partitionInput, i, start, end, numPartitions, std::ref(threadPartitions[i]));
        threads.push_back(std::move(thread));
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
// #include <iostream>
// #include <cmath>
// #include <thread>
// #include <vector>
// #include "tuples.h"
// #include <time.h>
// #include <chrono>

// using Partition = std::vector<std::tuple<int64_t, int64_t>>;
// tuple<int64_t, int64_t>* input;

// struct Worker {
//     int start, end, numPartitions;
//     std::vector<Partition> partitions;

//     Worker(int start, int end, int numPartitions)
//         : start(start), end(end), numPartitions(numPartitions), partitions(numPartitions) {}
    
//     void operator()() {
//         for (int i = start; i < end; i++) {
//             tuple<int64_t, int64_t> t = input[i];
//             int partitionKey = hashFunction(get<0>(t), numPartitions);
//             partitions[partitionKey].push_back(t);
//         }
//     }
// };

// void cleanup(std::vector<Worker>& workers) {
//     for (auto& worker : workers) {
//         for (auto& partition : worker.partitions) {
//             partition.clear();
//         }
//     }
// }

// int main(int argc, char* argv[]) {
//     const size_t numTuples = 16777216;
//     input = makeInput(numTuples);

//     const int numThreads = atoi(argv[1]);
//     const int numTuplesPerThread = numTuples / numThreads;

//     const int hashBits = atoi(argv[2]);
//     const int numPartitions = 1 << hashBits;
//     const int sizePartition = numTuples / numPartitions * 1.5;

//     std::vector<std::thread> threads;
//     std::vector<Worker> workers;
//     workers.reserve(numThreads);

//     for (int i = 0; i < numThreads; i++) {
//         auto start = i * numTuplesPerThread;
//         auto end = (i + 1) * numTuplesPerThread;
//         workers.emplace_back(start, end, numPartitions);
//     }

//     auto start_clock = std::chrono::steady_clock::now();

//     for (int i = 0; i < numThreads; i++) {
//         threads.emplace_back(std::thread(std::ref(workers[i])));
//     }

//     for (auto& t : threads) {
//         t.join();
//     }

//     auto end_clock = std::chrono::steady_clock::now();
//     std::chrono::duration<double> cpu_time_used = end_clock - start_clock;
//     printf("no threads %d no hash bits %d time:%f\n", numThreads, hashBits, cpu_time_used);

//     cleanup(workers);

//     return 0;
// }


