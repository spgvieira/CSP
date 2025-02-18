#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>

void *thread_function(void *arg) {
    printf("Start thread on CPU %d\n", sched_getcpu());
    while(1);
}

int main() {
    int core_count = 4;
    pthread_t threads[core_count];
    pthread_attr_t attr[core_count];
    cpu_set_t cpuset[core_count];

    for (int i = 0; i < core_count; i++) {
        // Initialize thread attributes
        pthread_attr_init(&attr[i]);

        // Set thread affinity to CPU i
        CPU_ZERO(&cpuset[i]);
        CPU_SET(i, &cpuset[i]);
        pthread_attr_setaffinity_np(&attr[i], sizeof(cpu_set_t), &cpuset[i]);

        // Create a new thread with specified affinity
        int result = pthread_create(&threads[i], &attr[i], thread_function, NULL);
    }

    // Wait for all threads to finish
    for (int i = 0; i < core_count; i++) {
        pthread_join(threads[i], NULL);
        pthread_attr_destroy(&attr[i]); // Clean up attributes
    }
    return 0;
}
