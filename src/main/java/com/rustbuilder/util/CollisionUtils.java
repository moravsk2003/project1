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
            double minA = Double.MAX_VALUE;
            double maxA = -Double.MAX_VALUE;
            for (int p = 0; p < polyA.length / 2; p++) {
                double dot = polyA[p * 2] * normalX + polyA[p * 2 + 1] * normalY;
                if (dot < minA) minA = dot;
                if (dot > maxA) maxA = dot;
            }

            double minB = Double.MAX_VALUE;
            double maxB = -Double.MAX_VALUE;
            for (int p = 0; p < polyB.length / 2; p++) {
                double dot = polyB[p * 2] * normalX + polyB[p * 2 + 1] * normalY;
                if (dot < minB) minB = dot;
                if (dot > maxB) maxB = dot;
            }

            // Check for gap with tolerance for floating point errors
            double EPSILON = 0.001;
            if (maxA <= minB + EPSILON || maxB <= minA + EPSILON) {
                return true; // Separated
            }
        }
        return false;
    }
}
