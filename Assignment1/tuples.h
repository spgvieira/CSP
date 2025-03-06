#include <iostream>
#include <tuple>
#include <array>
#include <cstdint>
#include <cmath>
using namespace std;


// The code in this class was adpated from the following article
// https://www.geeksforgeeks.org/shuffle-a-given-array-using-fisher-yates-shuffle-algorithm/

// A utility function to swap to integers 
void swap (tuple<int64_t,int64_t> *a, tuple<int64_t,int64_t> *b) 
{ 
    tuple<int64_t,int64_t> temp = *a; 
    *a = *b; 
    *b = temp; 
} 

// A function to generate a random 
// permutation of arr[] 
void randomize (tuple<int64_t,int64_t> arr[], int n) 
{ 
    // Use a different seed value so that 
    // we don't get same result each time
    // we run this program 
    srand (time(NULL)); 
 
    // Start from the last element and swap 
    // one by one. We don't need to run for 
    // the first element that's why i > 0 
    for (int i = n - 1; i > 0; i--) 
    { 
        // Pick a random index from 0 to i 
        int j = rand() % (i + 1); 
 
        // Swap arr[i] with the element 
        // at random index 
        swap(&arr[i], &arr[j]); 
    } 
} 

tuple<int64_t,int64_t>* makeInput(size_t numTuples) {
    tuple<int64_t, int64_t>* input = new tuple<int64_t, int64_t>[numTuples];
    
    for (int i = 0; i <numTuples; i++) {
        input[i] = std::make_tuple(i,0);
    }

    randomize(input, numTuples);

    return input;
}

// // modular hash function
// int hashFunction(int64_t key, int numPartitions ) {
//     return key % numPartitions;
// }

// multiplicative hash function
// int multiplicativeHash(int64_t key, int numPartitions) {
//     const double A = 0.6180339887; // Knuth's constant
//     double fractionalPart = fmod(key * A, 1.0); 
//     return static_cast<int>(numPartitions * fractionalPart);
// }

int hashFunction(int64_t key, int hashBits) {
    const uint64_t A = 2654435769; // Large prime close to Knuth's constant * 2^32
    return (key * A) >> (32 - hashBits); 
}