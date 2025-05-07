package assignment2.Mandelbrot;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.stream.IntStream;

public class FunctionalSequentialMandelbrot {
    public static void main(String[] args) throws IOException {
        int size = args.length >= 1 ? Integer.parseInt(args[0]) : 200;
        long startTime = System.currentTimeMillis();
        generateAndSave(size, "mandelbrot.pbm");
        long estimatedTime = System.currentTimeMillis() - startTime;
        System.out.println(estimatedTime);
    }

    private static void generateAndSave(int size, String filename) throws IOException {
        double[] coordinates = IntStream.range(0, size)
                .mapToDouble(i -> (double) i * 2.0 / size)
                .toArray();

        byte[][] imageData = generateMandelbrotImage(size, coordinates);
        // writeToFile(size, imageData, filename);
    }

    private static byte[][] generateMandelbrotImage(int size, double[] coordinates) {
        return IntStream.range(0, size)
                .mapToObj(y -> generateRow(y, size, coordinates))
                .toArray(byte[][]::new);
    }

    private static byte[] generateRow(int y, int size, double[] coordinates) {
    double ci = coordinates[y] - 1.0;

    return IntStream.range(0, (size + 7) / 8)
            .mapToObj(xb -> (byte) (computeByte(xb * 8, y, ci, coordinates, size) ^ 0xFF))
            .collect(
                ByteArrayOutputStream::new,
                (bos, b) -> bos.write(b),
                (bos1, bos2) -> {
                    try {
                        bos2.writeTo(bos1);
                    } catch (IOException e) {
                        throw new UncheckedIOException(e);
                    }
                }
            ).toByteArray();
    }

    private static byte computeByte(int x, int y, double ci, double[] coordinates, int size) {
        return (byte) computeBits(x, y, ci, coordinates, size, 0, 0);
    }
    
    private static int computeBits(int x, int y, double ci, double[] coordinates, int size, int i, int acc) {
        if (i >= 8) return acc;
    
        int pixelIndex = x + i;
        if (pixelIndex >= size) return acc << (8 - i); // pad remaining bits with 0s
    
        double cr = coordinates[pixelIndex] - 1.5;
        boolean escaped = hasEscaped(cr, ci, cr, ci, 49);
    
        return computeBits(x, y, ci, coordinates, size, i + 1, (acc << 1) | (escaped ? 1 : 0));
    }
    
    private static boolean hasEscaped(double cr, double ci, double zr, double zi, int iter) {
        if (iter <= 0) return false;
        if (zr * zr + zi * zi > 4.0) return true;
    
        double temp = zr * zr - zi * zi + cr;
        double newZi = 2 * zr * zi + ci;
        return hasEscaped(cr, ci, temp, newZi, iter - 1);
    }

    // private static void writeToFile(int size, byte[][] imageData, String filename) throws IOException {
    //     try (BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(filename))) {
    //         out.write(String.format("P4\n%d %d\n", size, size).getBytes());
    //         for (byte[] row : imageData) {
    //             out.write(row);
    //         }
    //     }
    //     System.out.println("Mandelbrot set image saved to " + filename);
    // }
}
