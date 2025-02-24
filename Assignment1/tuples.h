#include <iostream>
#include <tuple>
#include <array>
using namespace std;


// The code in this class was adpated from the following article
// https://www.geeksforgeeks.org/shuffle-a-given-array-using-fisher-yates-shuffle-algorithm/

// A utility function to swap to integers 
void swap (tuple<int8_t,int8_t> *a, tuple<int8_t,int8_t> *b) 
{ 
    tuple<int8_t,int8_t> temp = *a; 
    *a = *b; 
    *b = temp; 
} 

// A function to generate a random 
// permutation of arr[] 
void randomize (tuple<int8_t,int8_t> arr[], int n) 
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

// num_Tuples = 2^24
tuple<int8_t,int8_t>* makeInput(size_t numTuples) {
    tuple<int8_t, int8_t>* input = new tuple<int8_t, int8_t>[numTuples];
    
    for (int i = 0; i <numTuples; i++) {
        input[i] = std::make_tuple(i,0);
    }

    randomize(input, numTuples);

    return input;
}

// modular hash function
int hashFunction(int8_t key, int numPartitions ) {
    return key % numPartitions;
}
