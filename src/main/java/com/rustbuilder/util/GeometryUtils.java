package com.rustbuilder.util;

/**
 * Shared geometry helpers used across controllers and UI layers.
 *
 * <p>Centralises the ray-casting point-in-polygon test that was previously
 * duplicated in {@code GameController.isPointInPolygon()} and
 * {@code GameCanvas.contains()}.
 */
public final class GeometryUtils {

    private GeometryUtils() {}

    /**
     * Checks whether the point {@code (x, y)} lies inside the polygon
     * described by the flat coordinate array {@code poly}.
     *
     * <p>The array is laid out as {@code [x0, y0, x1, y1, …]} — i.e. pairs
     * of (x, y) values for each vertex.
     *
     * <p>Algorithm: even-odd ray-casting (parity of crossings).
     *
     * @param x    test point x
     * @param y    test point y
     * @param poly flat array of polygon vertices ({@code length = 2 * numVertices})
     * @return {@code true} if the point is inside the polygon
     */
    public static boolean isPointInPolygon(double x, double y, double[] poly) {
        boolean inside = false;
        int numVertices = poly.length / 2;
        for (int i = 0, j = numVertices - 1; i < numVertices; j = i++) {
            double xi = poly[i * 2], yi = poly[i * 2 + 1];
            double xj = poly[j * 2], yj = poly[j * 2 + 1];

            boolean intersect = ((yi > y) != (yj > y))
                    && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;
        }
        return inside;
    }
}
