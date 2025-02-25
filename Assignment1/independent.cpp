#include <iostream>
#include <cmath>
#include <thread>
#include <vector>
#include "tuples.h"

// TODO: 
// re read code and clean not used
// re evaluate which variables should be public
// change parameters to input values


tuple<int64_t, int64_t>* input;

void partitionInput(int numThread, int start, int end, int numPartitions, vector<vector<tuple<int64_t, int64_t>>>* partitions) {
    for (int i = start; i < end; i++) {
        tuple<int64_t, int64_t> t = input[i];
        int partitionKey = hashFunction(get<0>(t), numPartitions);

        if (partitionKey >= 0 && partitionKey < numPartitions) {
            (*partitions)[partitionKey].push_back(t);
        }
    }

    for (int p = 0; p < numPartitions; ++p) {
        cout << "Thread " << numThread << " - Partition " << p << ": ";
        for (const auto& tup : (*partitions)[p]) {
            cout << "(" << get<0>(tup) << ", " << get<1>(tup) << ") ";
        }
        cout << "\n";
    }
}


// void partitionInput(int numThread, int start, int end, int numPartitions, vector<vector<tuple<int64_t, int64_t>>>& partitions) {

//     for (int i = start; i < end; i++) {
//         tuple<int64_t, int64_t> t = input[i];
//         int partitionKey = hashFunction(get<0>(t), numPartitions);
//         partitions[partitionKey].push_back(t);
//     }

//     for (int i = 0; i < numPartitions; ++i) {
//         std::cout << "Array " << i << ": ";
//         for (size_t j = 0; j < partitions[i].size(); ++j) {
//             cout << get<0>(partitions[i][j]) << " ";
//         }
//         std::cout << "\n";
//     }
// }

int main() {
    const size_t numTuples = 10;
    input = makeInput(numTuples);

    for (size_t j = 0; j < numTuples; ++j) {
        cout << get<0>(input[j]) << " ";
    }
    cout << endl;

    const int numThreads = 2;
    const int numTuplesPerThread = numTuples / numThreads;

    const int hashBits = 1;
    const int numPartitions = pow(2, hashBits);

    std::vector<std::thread> threads;
    std::vector<std::vector<std::vector<std::tuple<int64_t, int64_t>>>> threadPartitions(numThreads, std::vector<std::vector<std::tuple<int64_t, int64_t>>>(numPartitions));

    // std::vector<std::vector<std::tuple<int64_t, int64_t>>> partitions(numPartitions);

    for (int i = 0; i < numThreads; i++) {
        auto start = i * numTuplesPerThread;
        auto end = (i + 1) * numTuplesPerThread;

        // threads.emplace_back(partitionInput, i, start, end, numPartitions, std::ref(partitions));
        threads.emplace_back(partitionInput, i, start, end, numPartitions, &threadPartitions[i]);
    }

    for (auto& t : threads) {
        t.join();
    }

    return 0;
}
