#include <iostream>
#include <cmath>
#include "tuples.h"
#include "includes/thread_pool.hpp"

using namespace std;

// In the independent algorithm, each thread
// has a buffer for every single outputBuffers
// each thread reads 1/t of the input, each thread receives input[:,:] when initialized
// iterated throught the portion of the array, sends tuple key to hash function
// picks output buffer from result of hash function
// writes on buffer
// done!

// question: Is threadpool good idea?

void partition() {
    
}

int main() {
    const size_t numTuples = 10; // size_t can store the maximum size of a theoretically possible object of any type
    tuple<int64_t,int64_t>* input = makeInput(numTuples);

    cout << "arr[0]: " << get<0>(input[0]) << endl;
    cout << "arr[1]: " << get<0>(input[1]) << endl;
    cout << "arr[2]: " << get<0>(input[2]) << endl;

    const int numThreads = 1;
    const int numTuplesPerThread = numTuples/numThreads;

    const int hashBits = 2;
    const int outputBuffers = pow(2, hashBits);

    for (int i=0; i<numThreads; i++) {
        auto start = input + i*numTuplesPerThread;
        auto end = input + i*(numTuplesPerThread + 1); // - 1 ?
        // dataThread = 
        partition();
    }

    // thread_pool pool(numThreads);
    // pool.submit(partition);
    // pool.wait_for_tasks();
    
}