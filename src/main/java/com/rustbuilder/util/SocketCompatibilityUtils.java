package com.rustbuilder.util;

import com.rustbuilder.model.core.Socket;

/**
 * Shared edge-socket matching rules for placement, collision, stability, and rewards.
 */
public final class SocketCompatibilityUtils {
    private static final double PARALLEL_TOLERANCE_DEGREES = 2.0;

    private SocketCompatibilityUtils() {}

    public static boolean areEdgeSocketsConnected(Socket a, Socket b, double maxDistanceSq) {
        if (!areEdgesParallel(a, b)) {
            return false;
        }

        double dx = a.getX() - b.getX();
        double dy = a.getY() - b.getY();
        return dx * dx + dy * dy < maxDistanceSq;
    }

    public static boolean areEdgesParallel(Socket a, Socket b) {
        if (!isEdgeSocket(a) || !isEdgeSocket(b)) {
            return false;
        }

        double aAngle = tangentAngleDegrees(a);
        double bAngle = tangentAngleDegrees(b);
        if (Double.isNaN(aAngle) || Double.isNaN(bAngle)) {
            return false;
        }

        double diff = Math.abs(normalizeHalfTurn(aAngle - bAngle));
        return diff <= PARALLEL_TOLERANCE_DEGREES;
    }

    public static boolean isEdgeSocket(Socket socket) {
        return socket != null && socket.getSide() != 10;
    }

    private static double tangentAngleDegrees(Socket socket) {
        double rotation = socket.getRotation();
        switch (socket.getSide()) {
            case 0:
            case 2:
            case 6:
                return rotation;
            case 1:
            case 3:
                return rotation + 90.0;
            case 4:
                return rotation - 60.0;
            case 5:
                return rotation + 60.0;
            default:
                return Double.NaN;
        }
    }

    private static double normalizeHalfTurn(double angle) {
        double normalized = angle % 180.0;
        if (normalized < -90.0) {
            normalized += 180.0;
        } else if (normalized > 90.0) {
            normalized -= 180.0;
        }
        return normalized;
    }
}
