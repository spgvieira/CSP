package assignment2;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class FunctionalSequentialSpectralNorm {

    private static final NumberFormat formatter = new DecimalFormat("#.000000000");

    record UV(double[] u, double[] v) {}

    public static void main(String[] args) {
        int n = 5500;
        // if (args.length > 0) n = Integer.parseInt(args[0]);
        
        double result = approximate(n);
        System.out.println(formatter.format(result));
    }

    // by making it static I know I'm only using input variables
    public static double approximate(int n) {
        double[] uInitial = DoubleStream.generate(() -> 1.0).limit(n).toArray();
        double[] vInitial = DoubleStream.generate(() -> 0.0).limit(n).toArray();

        UV finalUV = IntStream.range(0, 10)
            .boxed()
            .reduce( // same as fold
                new UV(uInitial, vInitial), 
                (uv, i) -> {
                    double[] vNew = multiplyAtAv(n, uv.u());
                    double[] uNew = multiplyAtAv(n, vNew);
                    return new UV(uNew, vNew);
                },
                (a, b) -> b 
            );

        double[] u = finalUV.u();
        double[] v = finalUV.v();

        double vBv = IntStream.range(0, n).mapToDouble(i -> u[i] * v[i]).sum();
        double vv  = IntStream.range(0, n).mapToDouble(i -> v[i] * v[i]).sum();

        return Math.sqrt(vBv / vv);
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
