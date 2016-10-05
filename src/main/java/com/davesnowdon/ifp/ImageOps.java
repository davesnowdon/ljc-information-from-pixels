package com.davesnowdon.ifp;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.util.ArrayList;
import java.util.List;

/**
 * Useful operations on images represented as OpenCV matrices
 */
public class ImageOps {
    public static final String FACE_XML = "haarcascade_frontalface_default.xml";

    public static CascadeClassifier faceClassifier() {
        // TODO dsnowdon need to get abs path of XML
        return new CascadeClassifier(FACE_XML);
    }

    /**
     * Apply a classifier to an image and return a list of matched rectangles
     * @param clr
     * @param image
     * @return
     */
    public static List<Rect> applyClassifier(CascadeClassifier clr, Mat image) {
        final MatOfRect result = new MatOfRect();
        clr.detectMultiScale(image, result);
        return result.toList();
    }

    /**
     * Read an image from file into an OpenCV matrix
     *
     * @param filename
     * @return
     */
    public static Mat readImage(String filename) {
        return Imgcodecs.imread(filename);
    }

    /**
     * Write an OpenCV matrix to an file guessing format from file extension
     *
     * @param filename
     * @param image
     */
    public static void writeImage(String filename, Mat image) {
        Imgcodecs.imwrite(filename, image);
    }

    /**
     * Returns an openCV matrix with the same dimensions and type as the
     * input matrix. Useful since most openCV functions that don't modify a
     * matrix in-place expect a destination matrix to be supplied
     *
     * @param matrix
     * @return
     */
    public static Mat resultMatrix(Mat matrix) {
        return new Mat(matrix.rows(), matrix.cols(), matrix.type());
    }

    /**
     * Convert an openCV matrix representing a colour image to a grayscale
     * one of the same size
     *
     * @param input
     * @return
     */
    public static Mat toGrayscale(Mat input) {
        final Mat gray = new Mat(input.rows(), input.cols(), CvType.CV_8UC3);
        Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
        return gray;
    }

    /**
     * Convert an OpenCV matric in BGR format to the HSV colour space
     * @param input
     * @return
     */
    public static Mat toHsv(Mat input) {
        final Mat hsv = resultMatrix(input);
        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_BGR2HSV);
        return hsv;
    }

    /**
     * Resize image to desired width, preserving aspect ratio
     * @param image
     * @param newWidth
     * @return
     */
    public static Mat resizeByWidth(Mat image, int newWidth) {
        final int width = image.cols();
        final int height = image.rows();
        final double ratio = ((double) newWidth) / width;
        final int newHeight = (int) (height * ratio);
        final Mat result = new Mat(newHeight, newWidth, image.type());
        Imgproc.resize(image, result, new Size(newWidth, newHeight));
        return result;
    }

    /**
     * Resize image to desired height, preserving aspect ratio
     * @param image
     * @param newHeight
     * @return
     */
    public static Mat resizeByHeight(Mat image, int newHeight) {
        final int width = image.cols();
        final int height = image.rows();
        final double ratio = ((double) newHeight) / height;
        final int newWidth = (int) (width * ratio);
        final Mat result = new Mat(newHeight, newWidth, image.type());
        Imgproc.resize(image, result, new Size(newWidth, newHeight));
        return result;
    }

    /**
     * Apply gaussian blurring to the supplied image. sigma X & sigma Y are
     * calculated from the kernel size
     *
     * @param image
     * @param kernelSize
     * @return
     */
    public static Mat gaussianBlur(Mat image, int kernelSize) {
        final Mat result = resultMatrix(image);
        Imgproc.GaussianBlur(image, result, new Size(kernelSize, kernelSize), 0.0);
        return result;
    }

    /**
     * Return the variance of a single channel image matrix
     *
     * @param gray
     * @return
     */
    public static double matrixVariance(Mat gray) {
        final MatOfDouble mean = new MatOfDouble();
        final MatOfDouble stddev = new MatOfDouble();
        Core.meanStdDev(gray, mean, stddev);
        final double sd = stddev.toList().get(0);
        return sd * sd;
    }

    /**
     * Determine if an image is blurry using the variance of a Laplacian of
     * the grayscale image. From
     * http://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
     *
     * @param image
     * @param threshold
     * @return
     */
    public static boolean isImageBlurry(Mat image, double threshold) {
        final Mat gray = toGrayscale(image);
        final Mat laplacian = resultMatrix(gray);
        Imgproc.Laplacian(gray, laplacian, CvType.CV_64F);
        final double variance = matrixVariance(laplacian);
        return variance < threshold;
    }

    public static boolean isImageBlurry(Mat image) {
        return isImageBlurry(image, 100.0);
    }

    /**
     * Find the contours in an image which is assumed to be grayscale
     * http://docs.opencv.org/3.1.0/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a
     *
     * @param image
     * @return
     */
    public static List<MatOfPoint> findContours(Mat image) {
        final List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(image, contours, new Mat(),
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE);
        return contours;
    }


}
