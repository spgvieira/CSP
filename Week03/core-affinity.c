#define _GNU_SOURCE
#include <sched.h>
#include <stdlib.h>
#include <pthread.h>

void *do_stuff(void *args) {
    int core_id = *(int *)args;
    cpu_set_t  mask; // a mask for which threads can be used. e.g. if the first and second bits are 1, then cores 0 and 1 and be used.
    CPU_ZERO(&mask);
    CPU_SET(core_id, &mask); // set the bit corresponding to the desired thread to 1
    int result = sched_setaffinity(0, sizeof(mask), &mask); // 0 means current thread. Returns 0 on succes
    while (1);
}

int main(int argc, char **argv) {
    int core_count = 4;
    int *args = malloc(core_count * sizeof(int)); // create pointer to hold args passed to threads
    pthread_t threads[core_count];
    for (int i = 0; i < core_count; i++) {
        args[i] = i;
        pthread_create(&threads[i], NULL, do_stuff, args+i);
    }

    for (int i = 0; i < core_count; i++) {
        pthread_join(threads[i], NULL);
    }
    free(args); // remember to free
    return 0;
}