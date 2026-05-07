package com.rustbuilder.model.core;

public class Socket {
    public static final int CENTER_SIDE = 10;

    private double x;
    private double y;
    private double rotation; // The rotation a block attaching here should take
    private int side; // 0=North, 1=East, 2=South, 3=West, CENTER_SIDE=center

    public Socket(double x, double y, double rotation, int side) {
        this.x = x;
        this.y = y;
        this.rotation = rotation;
        this.side = side;
    }

    public double getX() {
        return x;
    }

    public double getY() {
        return y;
    }

    public double getRotation() {
        return rotation;
    }

    public int getSide() {
        return side;
    }

    public boolean isCenter() {
        return side == CENTER_SIDE;
    }
}
