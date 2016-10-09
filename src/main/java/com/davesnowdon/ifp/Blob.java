package com.davesnowdon.ifp;


import org.opencv.core.MatOfPoint;

public class Blob {
    private final MatOfPoint contour;

    private final Circle enclosedBy;

    public Blob(MatOfPoint contour, Circle enclosedBy) {
        this.contour = contour;
        this.enclosedBy = enclosedBy;
    }

    public MatOfPoint getContour() {
        return contour;
    }

    public Circle getEnclosedBy() {
        return enclosedBy;
    }
}
