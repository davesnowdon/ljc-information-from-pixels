package com.davesnowdon.ifp;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.IntStream;

/**
 * Launcher for OpenCV demos for "Information from Pixels" talk
 */
public class Main {
    public static final String OPTION_COMMAND = "command";

    public static final String OPTION_IMAGE = "image";

    public static final String OPTION_OUTPUT = "output";

    public static final String OPTION_KERNEL_SIZE = "kernel-size";

    public static final String OPTION_LOW = "low";

    public static final String OPTION_HIGH = "high";

    public static final String OPTION_CLASSIFIER = "classifier";

    public static final String OPTION_INTERMEDIATE = "intermediate";

    public static final Scalar OUTLINE_COLOUR = new Scalar(0.0, 255.0, 0.0);

    public static final Scalar CENTRE_COLOUR = new Scalar(255.0, 0.0, 0.0);

    public static final String FACE_XML = "src/main/resources/haarcascade_frontalface_default.xml";

    static Set<String> commands = new HashSet<>(Arrays.asList("classifier", "show", "find-blob", "find-faces", "find-line", "blur", "shapes"));

    public static void main(String[] argv) {
        Options options = new Options();
        options.addOption("c", OPTION_COMMAND, true, "Command to run, one of: " + commands);
        options.addOption("i", OPTION_IMAGE, true, "Input image filename");
        options.addOption("o", OPTION_OUTPUT, true, "Output image filename");
        options.addOption("k", OPTION_KERNEL_SIZE, true, "Kernel size");
        options.addOption("l", OPTION_LOW, true, "Comma separated triple for low end of range");
        options.addOption("h", OPTION_HIGH, true, "Comma separated triple for high end of range");
        options.addOption("d", OPTION_CLASSIFIER, true, "XML file to use as classifier");
        options.addOption("v", OPTION_INTERMEDIATE, false, "Write intermediate images to file");

        CommandLineParser parser = new DefaultParser();
        try {
            CommandLine line = parser.parse(options, argv);

            if (!line.hasOption(OPTION_COMMAND)) {
                throw new ParseException("Must specify command");
            }
            final String command = line.getOptionValue(OPTION_COMMAND);
            if (!commands.contains(command)) {
                throw new ParseException("Invalid command: " + command);
            }

            if (!line.hasOption(OPTION_IMAGE)) {
                throw new ParseException("Must specify input image");
            }
            final String imageFilename = line.getOptionValue(OPTION_IMAGE);

            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

            Mat image = ImageOps.readImage(imageFilename);

            Mat output = null;
            switch (command) {
                case "blur":
                    output = commandBlur(image, line);
                    break;

                case "classifier":
                    output = commandApplyClassifier(image, line);
                    break;

                case "find-blob":
                    output = commandFindBlob(image, line);
                    break;

                case "find-faces":
                    output = commandFindFaces(image, line);
                    break;

                case "find-line":
                    output = commandFindVerticalLine(image, line);
                    break;

                case "shapes":
                    output = commandShapes(image, line);
                    break;

                case "show":
                    output = commandShow(image, line);
                    break;
            }

            if (null != output) {
                if (line.hasOption(OPTION_OUTPUT)) {
                    final String outputFilename = line.getOptionValue(OPTION_OUTPUT);
                    System.out.println("Writing output image: "+outputFilename);
                    ImageOps.writeImage(outputFilename, output);
                }

                BufferedImage javaImage = Util.matrixToImage(output);
                Util.displayImage(command, javaImage);
            }

        } catch (ParseException e) {
            System.err.println("Failed to parse arguments  Reason: " + e.getMessage());
            HelpFormatter formatter = new HelpFormatter();
            formatter.printHelp("demo", options);
        }
    }


    /**
     * Demo reading an image using OpenCv, converting it to a java image and displaying it using Swing
     */
    public static Mat commandShow(Mat image, CommandLine line) {
        return image;
    }

    /**
     * Apply the specified classifier against the supplied image
     *
     * @param image
     * @param line
     * @return
     */
    private static Mat commandApplyClassifier(Mat image, CommandLine line) throws ParseException {
        if (!line.hasOption(OPTION_CLASSIFIER)) {
            throw new ParseException("Need to specify classifier filename");
        }
        String classifierFilename = line.getOptionValue(OPTION_CLASSIFIER);
        return applyClassifier(image, classifierFilename);
    }

    /**
     * Read an image and locate any faces
     */
    public static Mat commandFindFaces(Mat image, CommandLine line) throws ParseException {
        return applyClassifier(image, FACE_XML);
    }

    private static Mat applyClassifier(Mat image, String classifierFilename) throws ParseException {
        final Mat gray = ImageOps.toGrayscale(image);
        final CascadeClassifier faceClassifier = new CascadeClassifier(classifierFilename);
        List<Rect> faces = ImageOps.applyClassifier(faceClassifier, gray);
        System.out.println(Integer.toString(faces.size()) + " objects found");

        for (Rect face : faces) {
            Imgproc.rectangle(image, ImageOps.minPoint(face), ImageOps.maxPoint(face), OUTLINE_COLOUR, 2);
        }

        return image;
    }

    public static Mat commandFindBlob(Mat image, CommandLine line) throws ParseException {
        if (!line.hasOption(OPTION_LOW) || !line.hasOption(OPTION_HIGH)) {
            throw new ParseException("Need to specify both low and high for range operations");
        }
        final String[] lows = line.getOptionValue(OPTION_LOW).split(",");
        if (3 != lows.length) {
            throw new ParseException("Low values should be <H>,<S>,<V>");
        }
        Scalar low = new Scalar(Double.valueOf(lows[0]), Double.valueOf(lows[1]), Double.valueOf(lows[2]));

        final String[] highs = line.getOptionValue(OPTION_HIGH).split(",");
        if (3 != highs.length) {
            throw new ParseException("High values should be <H>,<S>,<V>");
        }
        Scalar high = new Scalar(Double.valueOf(highs[0]), Double.valueOf(highs[1]), Double.valueOf(highs[2]));

        Optional<Blob> maybeBlob = ImageOps.findBlob(image, low, high);
        if (maybeBlob.isPresent()) {
            Imgproc.circle(image, maybeBlob.get().getEnclosedBy().getCentre(), 5, CENTRE_COLOUR, 2);
            Imgproc.drawContours(image, Arrays.asList(maybeBlob.get().getContour()), 0, OUTLINE_COLOUR, 2);
            return image;
        } else {
            return null;
        }
    }

    public static Mat commandBlur(Mat image, CommandLine line) throws ParseException {
        int kernelSize = 3;
        if (line.hasOption(OPTION_KERNEL_SIZE)) {
            kernelSize = Integer.parseInt(line.getOptionValue(OPTION_KERNEL_SIZE));
        }

        if ((kernelSize % 2) == 0) {
            throw new ParseException("Kernel size must be an odd number");
        }

        return ImageOps.gaussianBlur(image, kernelSize);
    }

    public static Mat commandShapes(Mat image, CommandLine line) {
        Mat gray = ImageOps.toGrayscale(image);
        Mat blurred = ImageOps.gaussianBlur(gray, 5);
        Mat edges = ImageOps.resultMatrix(blurred);
        Imgproc.Canny(blurred, edges, 75, 200);
        List<MatOfPoint> contours = ImageOps.sortContours(ImageOps.findContours(edges));

        System.out.println("num contours = " + contours.size());

        // Start at largest contour and look for 4-sided shapes
        for (MatOfPoint contour : contours) {
            final MatOfPoint2f m2f = new MatOfPoint2f();
            // need to convert the contour from a MatOfPoint to MatOfPoint2f
            m2f.fromList(contour.toList());

            /*
             * Produce an approximation of the polygon so that edges that are not-quite straight lines
             * get approximated to a straight line.
             */
            double perimeter = Imgproc.arcLength(m2f, true);
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(m2f, approx, 0.01 * perimeter, true);

            int numSides = approx.toList().size();
            if (4 == numSides) {
                Rect r = Imgproc.boundingRect(contour);
                Imgproc.drawContours(image, Arrays.asList(contour), -1, OUTLINE_COLOUR, 2);
                Imgproc.putText(image, Integer.toString(numSides), new Point(r.x, r.y - 10), 0, 0.5, OUTLINE_COLOUR, 2);
                break;
            }
        }
        return image;
    }

    /**
     * Detect a line in an image, Return [offset, orientation] or nil, if no
     line detected.  offset: rough position of the line on screen [-1,
     +1] (-1: on the extreme left, 1: on the extreme right, 0: centered)
     orientation: its orientation [-pi/2,pi/2] Adapted from the python
     implementation by Alexandre Mazel, https://youtu.be/UGj3H6ETHJg"
     * @param image
     * @param line
     * @return
     */
    public static Mat commandFindVerticalLine(Mat image, CommandLine line) {
        Mat gray = ImageOps.toGrayscale(image);
        writeIntermediateOutput(gray, "-1-gray", line);

        // Make a kernel to detect vertical lines
        Mat kernel = new Mat(1, 3, CvType.CV_64F);
        double[] kernel_values = {-1.0, 2.0, -1.0};
        kernel.put(0, 0, kernel_values);

        // convolve grayscale image with kernel
        Mat convolved = ImageOps.resultMatrix(gray);
        Imgproc.filter2D(gray, convolved, -1, kernel);
        writeIntermediateOutput(convolved, "-2-convolved", line);

        // threshold result
        Mat thresh = ImageOps.resultMatrix(gray);
        Imgproc.threshold(convolved, thresh, 45.0, 255, Imgproc.THRESH_TOZERO);
        writeIntermediateOutput(thresh, "-3-thresh", line);

        // Get the X position of the highest value pixel in each row
        int[] positions = ImageOps.argMaxRow(thresh);
        System.out.println("Row argmax ="+Arrays.toString(positions));

        // find rows with line segments
        int[] nonZeroPositions = IntStream.of(positions).filter(v -> v > 0).toArray();

        if (nonZeroPositions.length < 4) {
            System.out.println("Detected line is very short");
            return thresh;
        }

        /*
         * Sample the X positions of the line at the top, middle and bottom of the image and
         * use this to determine the line position and orientation
         */
        int firstNonZero = nonZeroPositions[0];
        int lastNonZero = nonZeroPositions[nonZeroPositions.length-1];
        int heightSampling = lastNonZero - firstNonZero;
        int samplingSize = Math.max(Math.min(nonZeroPositions.length / 40, 8), 1);
        System.out.println("Height sampling = "+heightSampling+", sampling size = "+samplingSize);

        final int len = nonZeroPositions.length;
        int[] topSamples = Arrays.copyOfRange(nonZeroPositions, 0, samplingSize);
        int[] middleSamples = Arrays.copyOfRange(nonZeroPositions, len/2, len/2+samplingSize);
        int[] bottomSamples = Arrays.copyOfRange(nonZeroPositions, len-samplingSize-1, len-1);

        double top = IntStream.of(topSamples).average().getAsDouble();
        double middle = IntStream.of(middleSamples).average().getAsDouble();
        double bottom = IntStream.of(bottomSamples).average().getAsDouble();
        System.out.println("Top = "+top+", middle = "+middle+", bottom = "+bottom);

        /*
         * Line horizontal location and orientation
         */
        double orientation = (top - bottom) / heightSampling;
        double offset = (middle / image.cols()) * 2 - 1;
        System.out.println("X offset = "+offset+", orienation = "+orientation);

        /*
         * Draw a straight line over the detected line
         */
        final int halfWidth = image.cols() / 2;
        final int halfHeight = image.rows() / 2;
        int x = halfWidth + (int) Math.round(offset * halfWidth);
        int xOffset = (int) Math.round(Math.sin(orientation) * halfHeight);
        System.out.println("X = "+x+", X orientation offset = "+xOffset);
        // Line with orientation
        Imgproc.line(image, new Point(x-xOffset, 0), new Point(x+xOffset, image.rows()-1), CENTRE_COLOUR, 2);

        // Line without orientation
        Imgproc.line(image, new Point(x, 0), new Point(x, image.rows()-1), OUTLINE_COLOUR, 2);


        return image;
    }

    private static void writeIntermediateOutput(Mat image, String suffix, CommandLine line) {
        boolean saveIntermediate = line.hasOption(OPTION_INTERMEDIATE) && line.hasOption(OPTION_OUTPUT);
        if (!saveIntermediate) {
            return;
        }

        String baseOutputFile = line.getOptionValue(OPTION_OUTPUT);

        int dot = baseOutputFile.lastIndexOf('.');
        String base = (dot == -1) ? baseOutputFile : baseOutputFile.substring(0, dot);
        String extension = (dot == -1) ? "jpg" : baseOutputFile.substring(dot+1);

        String filename = base + suffix + "." + extension;
        System.out.println("Writing intermediate image: "+filename);
        ImageOps.writeImage(filename, image);
    }
}
