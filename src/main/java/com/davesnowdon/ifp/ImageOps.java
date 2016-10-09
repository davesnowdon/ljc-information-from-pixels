package com.davesnowdon.ifp;


import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Useful operations on images represented as OpenCV matrices
 */
public class ImageOps {

    /**
     * Apply a classifier to an image and return a list of matched rectangles
     *
     * @param clr
     * @param image
     * @return
     */
    public static List<Rect> applyClassifier(CascadeClassifier clr, Mat image) {
        final MatOfRect result = new MatOfRect();
        clr.detectMultiScale(image, result);
        return result.toList();
    }

    public static List<Rect> applyClassifierToRegionOfInterest(CascadeClassifier clr, Mat image, Rect rect) {
        final MatOfRect result = new MatOfRect();
        final Mat roi = regionOfInterest(image, rect);
        clr.detectMultiScale(roi, result);
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
     *
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
     *
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
     *
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
     * Erode image http://docs.opencv.org/2.4/doc/tutorials/imgproc/erosion_dilatation/erosion_dilatation.html
     *
     * @param image
     * @param kernelSize
     * @param numIterations
     * @return
     */
    public static Mat erode(Mat image, int kernelSize, int numIterations) {
        final Mat result = resultMatrix(image);
        final Mat se = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(kernelSize, kernelSize));
        Imgproc.erode(image, result, se, new Point(-1, -1), numIterations);
        return result;
    }

    public static Mat erode(Mat image) {
        return erode(image, 3, 1);
    }

    /**
     * Dilate image http://docs.opencv.org/2.4/doc/tutorials/imgproc/erosion_dilatation/erosion_dilatation.html
     *
     * @param image
     * @param kernelSize
     * @param numIterations
     * @return
     */
    public static Mat dilate(Mat image, int kernelSize, int numIterations) {
        final Mat result = resultMatrix(image);
        final Mat se = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(kernelSize, kernelSize));
        Imgproc.dilate(image, result, se, new Point(-1, -1), numIterations);
        return result;
    }

    public static Mat dilate(Mat image) {
        return dilate(image, 3, 1);
    }


    /**
     * Return a mask representing the pixels within the specified range
     *
     * @param image
     * @param low
     * @param high
     */
    public static Mat rangeMask(Mat image, Scalar low, Scalar high) {
        final Mat result = new Mat(image.rows(), image.cols(), CvType.CV_8UC3);
        Core.inRange(image, low, high, result);
        return result;
    }

    /**
     * Convert an image into a mask indicating the pixels containing values
     * within the supplied lower and upper bounds in HSV colour space
     *
     * @param image
     * @param low
     * @param high
     * @return
     */
    public static Mat hsvMask(Mat image, Scalar low, Scalar high) {
        Mat result = gaussianBlur(image, 11);
        result = toHsv(result);
        result = rangeMask(result, low, high);
        result = erode(result, 3, 2);
        return dilate(result, 3, 2);
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
     * Create a matrix representing the region of interest of a larger image
     * defined by an openCV rect
     *
     * @param image
     * @param rect
     */
    public static Mat regionOfInterest(Mat image, Rect rect) {
        return image.submat(rect);
    }

    /**
     * Returns the corner of a rectangle with the smallest x & y values
     *
     * @param rect
     * @return
     */
    public static Point minPoint(Rect rect) {
        return new Point(rect.x, rect.y);
    }

    /**
     * Returns the corner of a rectangle with the largest x & y values
     *
     * @param rect
     * @return
     */
    public static Point maxPoint(Rect rect) {
        return new Point(rect.x + rect.width, rect.y + rect.height);
    }

    /**
     * Returns absolute version of rectangle embedded in larger
     * rectangle. use this to convert a rectangle found by applying a
     * classifier to a ROI into coordinates relative to the original image
     *
     * @param main
     * @param embedded
     */
    public static Rect offsetRect(Rect main, Rect embedded) {
        return new Rect(
                main.x + embedded.x,
                main.y + embedded.y,
                embedded.width,
                embedded.height);
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

    public static MatOfPoint largestContour(List<MatOfPoint> contours) {
        return contours.stream()
                .max((c1, c2) -> (Imgproc.contourArea(c1) > Imgproc.contourArea(c2) ? 1 : -1))
                .get();

    }

    /**
     * Given a contour return a map containing the X & Y coordinates of the
     * centre and the radius
     *
     * @param contour
     * @return
     */
    public static Circle minEnclosingCircle(MatOfPoint contour) {
        final Point centre = new Point();
        final float[] radiusArray = new float[1];
        final MatOfPoint2f m2f = new MatOfPoint2f();
        // need to convert the contour from a MatOfPoint to MatOfPoint2f
        m2f.fromList(contour.toList());
        // now actually get the enclosing circle
        Imgproc.minEnclosingCircle(m2f, centre, radiusArray);
        return new Circle(centre, radiusArray[0]);
    }

    /**
     * Return the centre and radius of the largest blob (if any) which lies
     * within the range low-high in HSV colour space. Adapted from python
     * implementation described in
     * http://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
     * Ideally we would compute the centre using moments (as in the python
     * example but the Moments class seems to have disapeared from openCV
     * 3.0.0, see github issue https://github.com/Itseez/opencv/issues/5017
     *
     * @param image
     * @param low
     * @param high
     * @return
     */
    public static Optional<Blob> findBlob(Mat image, Scalar low, Scalar high) {
        final Mat mask = hsvMask(image, low, high);
        MatOfPoint contour = largestContour(findContours(mask));
        if (null != contour) {
            return Optional.of(new Blob(contour, minEnclosingCircle(contour)));
        }
        return Optional.empty();
    }
}
