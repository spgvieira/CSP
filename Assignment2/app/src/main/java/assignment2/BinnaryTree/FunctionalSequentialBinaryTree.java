package assignment2.BinnaryTree;

import java.util.stream.IntStream;

public class FunctionalSequentialBinaryTree {

    public static void main(String[] args) {
        int n = (args.length > 0) ? Integer.parseInt(args[0]) : 21;
        int maxDepth = Math.max(6, n);
        int stretchDepth = maxDepth + 1;

        System.out.println("stretch tree of depth " + stretchDepth +
                "\t check: " + checkTree(createTree(stretchDepth)));

        trees(maxDepth);

        TreeNode longLivedTree = createTree(maxDepth);
        System.out.println("long lived tree of depth " + maxDepth +
                "\t check: " + checkTree(longLivedTree));
    }

    public static void trees(int maxDepth) {
        IntStream.iterate(4, d -> d + 2)
                .takeWhile(d -> d <= maxDepth)
                .forEach(depth -> {
                    int iterations = 16 << (maxDepth - depth);
                    int check = IntStream.range(0, iterations)
                            .map(i -> checkTree(createTree(depth)))
                            .sum();
                    System.out.println(iterations + "\t trees of depth " +
                            depth + "\t check: " + check);
                });
    }

    public static TreeNode createTree(int depth) {
        return (depth == 0) ? new TreeNode() :
                new TreeNode(createTree(depth - 1), createTree(depth - 1));
    }

    public static int checkTree(TreeNode node) {
        return (node.left == null) ? 1 :
                1 + checkTree(node.left) + checkTree(node.right);
    }

    public static class TreeNode {
        TreeNode left, right;

        public TreeNode() {
            this.left = null;
            this.right = null;
        }

        public TreeNode(TreeNode left, TreeNode right) {
            this.left = left;
            this.right = right;
        }
    }
}

