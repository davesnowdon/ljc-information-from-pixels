package com.davesnowdon.ifp;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * Launcher for OpenCV demos for "Information from Pixels" talk
 */
public class Main {
    public static void main(String[] argv) {
        try {
            BufferedImage image = ImageIO.read(new File("/home/dns/Downloads/xmas2007-1.jpg"));
            Util.displayImage("demo", image);
        } catch (IOException e) {
            System.out.println("Failed to read image");
            System.exit(1);
        }
    }
}
