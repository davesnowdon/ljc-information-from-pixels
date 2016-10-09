package com.davesnowdon.ifp;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Launcher for OpenCV demos for "Information from Pixels" talk
 */
public class Main {
    public static final String OPTION_COMMAND = "command";

    public static final String OPTION_IMAGE = "image";

    public static final String OPTION_OUTPUT = "output";

    public static final String OPTION_KERNEL_SIZE = "kernel-size";

    public static final Scalar RED = new Scalar(255.0, 0.0, 0.0);

    public static final Scalar GREEN = new Scalar(0.0, 255.0, 0.0);

    public static final Scalar BLUE = new Scalar(255.0, 0.0, 0.0);

    public static final String FACE_XML = "src/main/resources/haarcascade_frontalface_default.xml";

    static Set<String> commands = new HashSet<>(Arrays.asList("show", "find-blob", "find-faces", "blur"));

    public static void main(String[] argv) {
        Options options = new Options();
        options.addOption("c", OPTION_COMMAND, true, "Command to run, one of: " + commands);
        options.addOption("i", OPTION_IMAGE, true, "Input image filename");
        options.addOption("o", OPTION_OUTPUT, true, "Output image filename");
        options.addOption("k", OPTION_KERNEL_SIZE, true, "Kernel size");

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
                case "show":
                    output = commandShow(image, line);
                    break;

                case "find-faces":
                    output = commandFindFaces(image, line);
                    break;

                case "find-blob":
                    output = commandFindBlob(image, line);
                    break;

                case "blur":
                    output = commandBlur(image, line);
                    break;
            }

            if (null != output) {
                if (line.hasOption(OPTION_OUTPUT)) {
                    final String outputFilename = line.getOptionValue(OPTION_OUTPUT);
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
     * Read an image and locate any faces
     */
    public static Mat commandFindFaces(Mat image, CommandLine line) {
        final Mat gray = ImageOps.toGrayscale(image);
        final CascadeClassifier faceClassifier = new CascadeClassifier(FACE_XML);
        List<Rect> faces = ImageOps.applyClassifier(faceClassifier, gray);
        System.out.println(Integer.toString(faces.size()) + " faces found");

        for (Rect face : faces) {
            Imgproc.rectangle(image, ImageOps.minPoint(face), ImageOps.maxPoint(face), BLUE, 2);
        }

        return image;
    }

    public static Mat commandFindBlob(Mat image, CommandLine line) throws ParseException {
        return null;
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
}
