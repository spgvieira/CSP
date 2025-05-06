
package assignment2.SepctralNorm;
/*
The Computer Language Benchmarks Game
https://salsa.debian.org/benchmarksgame-team/benchmarksgame/

contributed by Ziad Hatahet
based on the Go entry by K P anonymous
*/

// JAVA NAOT #3 (closest in performance that does not use cyclic barriers)
// https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/spectralnorm-graalvmaot-3.html
import java.text.DecimalFormat;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public class ImperativeParallelSpectralNorm {
    private static final DecimalFormat formatter = new DecimalFormat("#.000000000");

    public static void main(String[] args) throws InterruptedException {
        final int n = args.length > 0 ? Integer.parseInt(args[0]) : 5500;
        final int nThreads = args.length > 1 ? Integer.parseInt(args[1]) : 1;
        final ForkJoinPool customPool = new ForkJoinPool(nThreads);
        long startTime = System.currentTimeMillis();
        final var u = new double[n];
        for (int i = 0; i < n; i++)
            u[i] = 1.0;
        final var v = new double[n];
        for (int i = 0; i < 10; i++) {
            aTimesTransp(v, u, n, customPool);
            aTimesTransp(u, v, n, customPool);
        }

        double vBv = 0.0, vv = 0.0;
        for (int i = 0; i < n; i++) {
            final var vi = v[i];
            vBv += u[i] * vi;
            vv += vi * vi;
        }
        
        // System.out.println(formatter.format(Math.sqrt(vBv / vv)));
        double result = Math.sqrt(vBv / vv);
        formatter.format(result);
        // System.out.println(formatter.format(result));
        long estimatedTime = System.currentTimeMillis() - startTime;
        System.out.println(estimatedTime);
    }

    private static void aTimesTransp(double[] v, double[] u, int n, ForkJoinPool pool) {
        final var x = new double[n];
        pool.invoke(new TimesTask(x, 0, n, u, false));
        pool.invoke(new TimesTask(v, 0, n, x, true));
    }

     private static class TimesTask extends RecursiveAction {
        private final double[] v, u;
        private final int ii, n;
        private final boolean transpose;

        public TimesTask(double[] v, int ii, int n, double[] u, boolean transpose) {
            this.v = v;
            this.u = u;
            this.ii = ii;
            this.n = n;
            this.transpose = transpose;
        }

        @Override
        protected void compute() {
            if (n - ii <= 1) { 
                final var ul = u.length;
                double vi = 0.0;
                for (int j = 0; j < ul; j++) {
                    if (transpose)
                        vi += u[j] / a(j, ii);
                    else
                        vi += u[j] / a(ii, j);
                }
                v[ii] = vi;
            } else {
                int mid = ii + (n - ii) / 2;
                TimesTask left = new TimesTask(v, ii, mid, u, transpose);
                TimesTask right = new TimesTask(v, mid, n, u, transpose);
                invokeAll(left, right);
            }
        }

        private static int a(int i, int j) {
            return (i + j) * (i + j + 1) / 2 + i + 1;
        }
    }
}