package assignment2.Mandelbrot;

/* The Computer Language Benchmarks Game
 * https://salsa.debian.org/benchmarksgame-team/benchmarksgame/
 * 
 * contributed by Stefan Krause
 * slightly modified by Chad Whipkey
 * parallelized by Colin D Bennett 2008-10-04
 * reduce synchronization cost by The Anh Tran
 * optimizations and refactoring by Enotus 2010-11-11
 */

 // JAVA NAOT #6
 // https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/mandelbrot-graalvmaot-6.html

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public class ImperativeParallelMandelbrot {

    static byte[][] out;
    static double[] Crb;
    static double[] Cib;

    static int getByte(int x, int y){
        double Ci = Cib[y];
        int res = 0;
        for (int i = 0; i < 8; i += 2) {
            double Zr1 = Crb[x + i];
            double Zi1 = Ci;

            double Zr2 = Crb[x + i + 1];
            double Zi2 = Ci;

            int b = 0;
            int j = 49;
            do {
                double nZr1 = Zr1 * Zr1 - Zi1 * Zi1 + Crb[x + i];
                double nZi1 = Zr1 * Zi1 + Zr1 * Zi1 + Ci;
                Zr1 = nZr1;
                Zi1 = nZi1;

                double nZr2 = Zr2 * Zr2 - Zi2 * Zi2 + Crb[x + i + 1];
                double nZi2 = Zr2 * Zi2 + Zr2 * Zi2 + Ci;
                Zr2 = nZr2;
                Zi2 = nZi2;

                if (Zr1 * Zr1 + Zi1 * Zi1 > 4) b |= 2;
                if (Zr2 * Zr2 + Zi2 * Zi2 > 4) b |= 1;
                if (b == 3) break;
            } while (--j > 0);
            res = (res << 2) + b;
        }
        return res ^ -1;
    }

    static void putLine(int y, byte[] line){
        for (int xb = 0; xb < line.length; xb++)
            line[xb] = (byte) getByte(xb * 8, y);
    }

    static class MandelbrotTask extends RecursiveAction {
        private final int startY, endY;
        private static final int THRESHOLD = 64;

        MandelbrotTask(int startY, int endY) {
            this.startY = startY;
            this.endY = endY;
        }

        @Override
        protected void compute() {
            if (endY - startY <= THRESHOLD) {
                for (int y = startY; y < endY; y++) {
                    putLine(y, out[y]);
                }
            } else {
                int mid = (startY + endY) / 2;
                invokeAll(new MandelbrotTask(startY, mid), new MandelbrotTask(mid, endY));
            }
        }
    }

    public static void main(String[] args) throws Exception {
        int N = 6000;
        if (args.length >= 1) N = Integer.parseInt(args[0]);
        int threads = args.length >= 2 ? Integer.parseInt(args[1]) : Runtime.getRuntime().availableProcessors();

        long startTime = System.currentTimeMillis();
        Crb = new double[N + 7];
        Cib = new double[N + 7];
        double invN = 2.0 / N;
        for (int i = 0; i < N; i++) {
            Cib[i] = i * invN - 1.0;
            Crb[i] = i * invN - 1.5;
        }

        out = new byte[N][(N + 7) / 8];

        ForkJoinPool pool = new ForkJoinPool(threads);
        pool.invoke(new MandelbrotTask(0, N));

        // OutputStream stream = new BufferedOutputStream(System.out);
        // stream.write(("P4\n" + N + " " + N + "\n").getBytes());
        // for (int i = 0; i < N; i++) stream.write(out[i]);
        // stream.close();
        
        long estimatedTime = System.currentTimeMillis() - startTime;
        System.out.println(estimatedTime);
    }
}

