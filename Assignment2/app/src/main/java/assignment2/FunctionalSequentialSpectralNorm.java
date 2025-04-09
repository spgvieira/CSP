package assignment2;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class FunctionalSequentialSpectralNorm {

    private static final NumberFormat formatter = new DecimalFormat("#.000000000");

    public static void main(String[] args) {
        int n = 100;
        if (args.length > 0) n = Integer.parseInt(args[0]);
        
        double result = approximate(n);
        System.out.println(formatter.format(result));
    }

    // by making it static I know I'm only using input variables
    private static double approximate(int n) {
        // create unit vector
        double[] u = DoubleStream.generate(() -> 1.0).limit(n).toArray();
    
        return 0.0;
    }

    /* return element i,j of infinite matrix A */
    private static double A(int i, int j){
        return 1.0/((i+j)*(i+j+1)/2 +i+1);
    }

    /* multiply vector v by matrix A */
    private static double[] multiplyAv(int n, double[] v) {
        return IntStream.range(0, n)
                .mapToDouble(i ->
                        IntStream.range(0, n)
                                .mapToDouble(j -> A(i, j) * v[j])
                                .sum())
                .toArray();
    }

    /* multiply vector v by matrix A transposed */
    private static double[] multiplyAtv(int n, double[] v) {
        return IntStream.range(0, n)
                .mapToDouble(i ->
                        IntStream.range(0, n)
                                .mapToDouble(j -> A(j, i) * v[j])
                                .sum())
                .toArray();
    }

    /* multiply vector v by matrix A and then by matrix A transposed */
    private static double[] multiplyAtAv(int n, double[] v) {
        return multiplyAtv(n, multiplyAv(n, v));
    }
    
}
