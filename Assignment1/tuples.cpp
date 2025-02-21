// I have a list of tuples (key, value)
// I will send the tuple to the hash function
// The hash function will determine which partition the tuple will go to based
// on the key

#include <iostream>
#include <tuple>
#include <array>
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

int main() {
    const size_t numTuples = 10; // std::size_t can store the maximum size of a theoretically possible object of any type 
    std::tuple<int64_t,int64_t> input[numTuples];
    
    for (int i = 0; i <numTuples; i++) {
        input[i] = std::make_tuple(i,0);
    }

    cout << "arr[0]: " << get<0>(input[0]) << endl;
    cout << "arr[1]: " << get<0>(input[1]) << endl;
    cout << "arr[2]: " << get<0>(input[2]) << endl;

    randomize(input, numTuples);

    cout << "arr[0]: " << get<0>(input[0]) << endl;
    cout << "arr[1]: " << get<0>(input[1]) << endl;
    cout << "arr[2]: " << get<0>(input[2]) << endl;
}