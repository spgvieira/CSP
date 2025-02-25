#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int horizontal(u_int64_t size, u_int64_t *mem) {
  int x = 0;
  for (int rep = 0; rep < 1000; rep++) {
    for (u_int64_t i = 0; i < size; i++) {
      for (u_int64_t j = 0; j < size; j++) {
        mem[i * size + j] == 0 ? x++ : x--;
      }
    }
  }
  return x;
}

int vertical(u_int64_t size, u_int64_t *mem) {
  int x = 0;
  for (int rep = 0; rep < 1000; rep++) {
    for (u_int64_t i = 0; i < size; i++) {
      for (u_int64_t j = 0; j < size; j++) {
        mem[j * size + i] == 0 ? x++ : x--;
      }
    }
  }
  return x;
}

int main() {

  u_int64_t size = 10;
  u_int64_t max = 4000;

  printf("Testing up to array of size %lu\n", max * 2);

  clock_t start, end;
  double cpu_time_used;

  struct timespec begin, finish;

  int aux = 0;
  printf("clockspersec %i", CLOCKS_PER_SEC);
  for (u_int64_t c = size; c <= max; c += 250) {
    printf("Testing for array of size %lu * %lu = %lu\n", c, c, c * c);
    u_int64_t *arr = malloc(c * c * 8);

    start = clock();
    clock_gettime(CLOCK_MONOTONIC_RAW, &begin);
    aux += horizontal(c, arr);
    clock_gettime(CLOCK_MONOTONIC_RAW, &finish);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Horizontal took %f cpu seconds\n", cpu_time_used);

    start = clock();
    aux += vertical(c, arr);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Vertical took %f cpu seconds\n\n", cpu_time_used);
  }

  printf("%d", aux);
}
