package assignment2.FannkuchRedux;

import java.util.Arrays;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class FunctionalSequentialFannkuchRedux {
    
    public static int fannkuch(int n) {
        return permutations(n)
                .mapToInt(perm -> {
                    int flipsCount = 0;
                    int[] p = Arrays.copyOf(perm, n);
                    while (p[0] != 0) {
                        int k = p[0];
                        int k2 = (k + 1) >> 1;
                        for (int i = 0; i < k2; i++) {
                            int temp = p[i];
                            p[i] = p[k - i];
                            p[k - i] = temp;
                        }
                        flipsCount++;
                    }
                    return flipsCount;
                })
                .reduce(0, Math::max);
    }

    public static Stream<int[]> permutations(int n) {
        if (n == 0) {
            return Stream.of(new int[0]);
        }
        return permutations(n - 1)
                .flatMap(prevPermutation -> IntStream.rangeClosed(0, prevPermutation.length)
                        .mapToObj(i -> {
                            int[] permutation = new int[n];
                            System.arraycopy(prevPermutation, 0, permutation, 0, i);
                            permutation[i] = n - 1;
                            System.arraycopy(prevPermutation, i, permutation, i + 1, prevPermutation.length - i);
                            return permutation;
                        }));
    }

    public static void main(String[] args) {
        int n = 12;
        if (args.length > 0) {
            n = Integer.parseInt(args[0]);
        }
        int maxFlips = fannkuch(n);
        long checksum = calculateChecksum(n);
        System.out.println(checksum);
        System.out.println("Pfannkuchen(" + n + ") = " + maxFlips);
    }

    public static long calculateChecksum(int n) {
        IntStream flipsStream = permutations(n)
                .mapToInt(perm -> {
                    int flipsCount = 0;
                    int[] p = Arrays.copyOf(perm, n);
                    while (p[0] != 0) {
                        int k = p[0];
                        int k2 = (k + 1) >> 1;
                        for (int i = 0; i < k2; i++) {
                            int temp = p[i];
                            p[i] = p[k - i];
                            p[k - i] = temp;
                        }
                        flipsCount++;
                    }
                    return flipsCount;
                });
    
        return IntStream.range(0, (int) permutations(n).count()) // Generate indices
                .mapToLong(index -> {
                    int flips = flipsStream.skip(index).findFirst().orElse(0); // Get the corresponding flip count
                    return (index % 2 == 0) ? flips : -flips;
                })
                .sum();
    }
}

