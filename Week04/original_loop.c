#include <stdio.h> 
#include <stdlib.h>
#include <time.h> 

int main() {
    int start_size = 1000;
    long n = 1000000000;
    int64_t sum = 0;
    int *arr = NULL;
    clock_t start, end; 

    for (int size = start_size; size <= n; size *= 10) {
        arr = (int*) malloc (size * sizeof(int));

        for (int i = 0; i < size; i++) { 
            arr[i] = 1; 
        }

        start = clock();
        for (int i = 0; i < size; i++) 
        {
            sum += arr[i];
        }
        end = clock();
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

        printf("Size %d - The sum is %lld and took time %f ms\n", size, sum, time_taken); 

        free(arr);
    }

    return 0;
}