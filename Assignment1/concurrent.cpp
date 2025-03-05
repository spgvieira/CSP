#include <iostream>
#include <cmath>
#include <thread>
#include <mutex>
#include <vector>
#include "tuples.h"
#include <time.h>

// TODO: 
// re read code and clean not used
// re evaluate which variables should be public
// missing affinity

tuple<int64_t, int64_t>* input;

void partitionInput(int numThread, int start, int end, int numPartitions, vector<vector<tuple<int64_t, int64_t>>>& partitions, std::vector<std::mutex>& locks) {

    for (int i = start; i < end; i++) {
        tuple<int64_t, int64_t> t = input[i];
        int partitionKey = hashFunction(get<0>(t), numPartitions);
        locks[partitionKey].lock();
        partitions[partitionKey].push_back(t);
        locks[partitionKey].unlock();
    }

}

int main(int argc, char* argv[]) {
    // const size_t numTuples = 8;
    const size_t numTuples = 16777216;
    input = makeInput(numTuples);

    // for (size_t j = 0; j < numTuples; ++j) {
    //     cout << get<0>(input[j]) << " ";
    // }
    // cout << endl;

    const int numThreads = atoi(argv[1]);
    const int numTuplesPerThread = numTuples / numThreads;

    const int hashBits = atoi(argv[2]);
    const int numPartitions = pow(2, hashBits);
    const int sizePartition = numTuples/numPartitions * 1.5;

    std::vector<std::thread> threads(numThreads);
    std::vector<std::vector<std::tuple<int64_t, int64_t>>> partitions(numPartitions,std::vector<std::tuple<int64_t, int64_t>>(sizePartition));
    std::vector<std::mutex> locks(numPartitions);

    clock_t start, end;
    double cpu_time_used;

    start = clock();
    
    for (int i = 0; i < numThreads; i++) {
        auto start = i * numTuplesPerThread;
        auto end = (i + 1) * numTuplesPerThread;

        threads.emplace_back(partitionInput, i, start, end, numPartitions, std::ref(partitions), std::ref(locks));
    }

    for (auto& t : threads) {
        t.join();
    }

    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("no threads %d no hash bits %d time:%f\n", numThreads, hashBits, cpu_time_used);
    // for (int i = 0; i < numPartitions; ++i) {
    //     std::cout << "Array " << i << ": ";
    //     for (size_t j = 0; j < partitions[i].size(); ++j) {
    //         cout << get<0>(partitions[i][j]) << " ";
    //     }
    //     std::cout << "\n";
    // }

    return 0;
}
