package com.rustbuilder.util;

public class CollisionUtils {

    public static boolean checkCollision(double[] poly1, double[] poly2) {
        if (poly1.length == 0 || poly2.length == 0)
            return false; // No collision if no points

        // Check axes of poly1
        if (isSeparated(poly1, poly2))
            return false;
        // Check axes of poly2
        if (isSeparated(poly2, poly1))
            return false;

        return true;
    }

    private static boolean isSeparated(double[] polyA, double[] polyB) {
        int n = polyA.length / 2;
        double[] minMaxA = new double[2];
        double[] minMaxB = new double[2];
        
        for (int i = 0; i < n; i++) {
            // Edge vector
            double x1 = polyA[i * 2];
            double y1 = polyA[i * 2 + 1];
            double x2 = polyA[((i + 1) % n) * 2];
            double y2 = polyA[((i + 1) % n) * 2 + 1];

            double edgeX = x2 - x1;
            double edgeY = y2 - y1;

            // Normal (Perpendicular)
            double normalX = -edgeY;
            double normalY = edgeX;

            // Project both polygons onto normal
            project(polyA, normalX, normalY, minMaxA);
            project(polyB, normalX, normalY, minMaxB);

            // Check for gap with tolerance for floating point errors
            double EPSILON = 0.001;
            if (minMaxA[1] <= minMaxB[0] + EPSILON || minMaxB[1] <= minMaxA[0] + EPSILON) {
                return true; // Separated
            }
        }
        return false;
    }

    private static void project(double[] poly, double nx, double ny, double[] outMinMax) {
        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;

        for (int i = 0; i < poly.length / 2; i++) {
            double dot = poly[i * 2] * nx + poly[i * 2 + 1] * ny;
            if (dot < min)
                min = dot;
            if (dot > max)
                max = dot;
        }
        outMinMax[0] = min;
        outMinMax[1] = max;
    }
}
