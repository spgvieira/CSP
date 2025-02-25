#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

  printf("Testing up to array of size %llu\n", max * 2);

  clock_t start, end;
  double cpu_time_used;

  int aux = 0;

  for (u_int64_t c = size; c <= max; c += 250) {
    printf("Testing for array of size %llu * %llu = %llu\n", c, c, c * c);
    u_int64_t *arr = malloc(c * c * 8);

    start = clock();
    aux += vertical(c, arr);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Vertical took %f cpu milliseconds\n\n", cpu_time_used);
  }

  printf("%d", aux);
}
