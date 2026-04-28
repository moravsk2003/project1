package com.rustbuilder.util;

public class TransformUtils {

    /**
     * Rotates a point around a center.
     * 
     * @param x       The x coordinate of the point.
     * @param y       The y coordinate of the point.
     * @param cx      The x coordinate of the center of rotation.
     * @param cy      The y coordinate of the center of rotation.
     * @param degrees The angle in degrees.
     * @return A double array [x, y] containing the rotated coordinates.
     */
    public static double[] rotatePoint(double x, double y, double cx, double cy, double degrees) {
        double rad = Math.toRadians(degrees);
        double cos = Math.cos(rad);
        double sin = Math.sin(rad);

        double dx = x - cx;
        double dy = y - cy;

        double rx = dx * cos - dy * sin;
        double ry = dx * sin + dy * cos;

        return new double[] { cx + rx, cy + ry };
    }

    /**
     * Rotates a point around (0,0).
     * 
     * @param x       The x coordinate of the point.
     * @param y       The y coordinate of the point.
     * @param degrees The angle in degrees.
     * @return A double array [x, y] containing the rotated coordinates.
     */
    public static double[] rotatePoint(double x, double y, double degrees) {
        return rotatePoint(x, y, 0, 0, degrees);
    }
}
