package assignment2.FannkuchRedux;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class FunctionalSequentialFannkuchRedux {

        public static void main(String[] args) {
            int n = 12; // Change this value as needed
    
            int maxFlips = generatePermutations(IntStream.rangeClosed(1, n).boxed().collect(Collectors.toList()))
                .mapToInt(FunctionalSequentialFannkuchRedux ::countFlips)
                .max()
                .orElse(0);
    
            System.out.println("Max Flips: " + maxFlips);
        }
    
        // Generate all permutations functionally
        static Stream<List<Integer>> generatePermutations(List<Integer> list) {
            if (list.isEmpty()) {
                return Stream.of(Collections.emptyList());
            }
            return IntStream.range(0, list.size()).boxed()
                .flatMap(i -> {
                    List<Integer> rest = new ArrayList<>(list);
                    Integer head = rest.remove((int) i);
                    return generatePermutations(rest)
                        .map(perm -> {
                            List<Integer> newPerm = new ArrayList<>();
                            newPerm.add(head);
                            newPerm.addAll(perm);
                            return newPerm;
                        });
                });
        }
    
        // Count flips recursively
        static int countFlips(List<Integer> list) {
            if (list.get(0) == 1) return 0;
            List<Integer> flipped = new ArrayList<>(list);
            int k = flipped.get(0);
            Collections.reverse(flipped.subList(0, k));
            return 1 + countFlips(flipped);
        }
    
}