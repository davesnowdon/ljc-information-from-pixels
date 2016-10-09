package com.davesnowdon.ifp;


import org.opencv.core.Point;

public class Circle {
    private final Point centre;

    private final double radius;

    public Circle(Point centre, double radius) {
        this.centre = centre;
        this.radius = radius;
    }

    public Circle(double x, double y, double radius) {
        this(new Point(x, y), radius);
    }

    public Point getCentre() {
        return centre;
    }

    public double getRadius() {
        return radius;
    }
}
