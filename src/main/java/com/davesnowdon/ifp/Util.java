package com.davesnowdon.ifp;

import java.awt.*;
import java.awt.image.BufferedImage;

import org.opencv.core.Mat;

import javax.swing.*;

/**
 * Utilities and helper functions
 */
public class Util {
    /**
     * Create a java BufferedImage from an OpenCV matrix
     * @param mat
     * @return
     */
    public static BufferedImage matrixToImage(Mat mat) {
        final int width = mat.cols();
        final int height = mat.rows();
        final int depth = (int) mat.elemSize();
        final int type = (3 == depth) ? BufferedImage.TYPE_3BYTE_BGR : BufferedImage.TYPE_BYTE_GRAY;
        final byte[] bytes = new byte[width * height * depth];
        final BufferedImage image = new BufferedImage(width, height, type);
        mat.get(0, 0, bytes);
        image.getRaster().setDataElements(0, 0, width, height, bytes);
        return image;
    }

    /**
     * Display an image in a new frame
     * @param title
     * @param image
     */
    public static void displayImage(final String title, final BufferedImage image) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame(title);
            frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            ImageIcon imageIcon = new ImageIcon(image);
            JLabel jLabel = new JLabel();
            jLabel.setIcon(imageIcon);
            frame.getContentPane().add(jLabel, BorderLayout.CENTER);
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
}
