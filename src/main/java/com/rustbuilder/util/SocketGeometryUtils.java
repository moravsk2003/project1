package com.rustbuilder.util;

import java.util.ArrayList;
import java.util.List;

import com.rustbuilder.config.GameConstants;
import com.rustbuilder.model.core.Socket;

public class SocketGeometryUtils {

    public static List<Socket> getSquareSockets(double x, double y, double rotation) {
        List<Socket> sockets = new ArrayList<>();
        double size = GameConstants.TILE_SIZE;
        double cx = x + size / 2;
        double cy = y + size / 2;

        double rotRad = Math.toRadians(rotation);
        double cos = Math.cos(rotRad);
        double sin = Math.sin(rotRad);

        double half = size / 2;

        sockets.add(new Socket(cx + 0 * cos - (-half) * sin, cy + 0 * sin + (-half) * cos, rotation, 0));
        sockets.add(new Socket(cx + half * cos - 0 * sin, cy + half * sin + 0 * cos, rotation, 1));
        sockets.add(new Socket(cx + 0 * cos - half * sin, cy + 0 * sin + half * cos, rotation, 2));
        sockets.add(new Socket(cx + (-half) * cos - 0 * sin, cy + (-half) * sin + 0 * cos, rotation, 3));

        sockets.add(new Socket(cx, cy, rotation, 10));

        return sockets;
    }

    public static List<Socket> getTriangleSockets(double x, double y, double rotation) {
        List<Socket> sockets = new ArrayList<>();
        double size = GameConstants.TILE_SIZE;
        double cx = x + size / 2;
        double cy = y + size / 2;

        double rotRad = Math.toRadians(rotation);
        double cos = Math.cos(rotRad);
        double sin = Math.sin(rotRad);

        double[][] offsets = {
                { 0, GameConstants.HALF_TILE, 6 }, // Base
                { -GameConstants.HALF_TILE / 2, GameConstants.TRIANGLE_OFFSET / 2, 4 }, // Left Slope
                { GameConstants.HALF_TILE / 2, GameConstants.TRIANGLE_OFFSET / 2, 5 } // Right Slope
        };

        for (double[] off : offsets) {
            double ox = off[0];
            double oy = off[1];
            int side = (int) off[2];

            double rx = ox * cos - oy * sin;
            double ry = ox * sin + oy * cos;

            sockets.add(new Socket(cx + rx, cy + ry, rotation, side));
        }

        sockets.add(new Socket(cx, cy, rotation, 10));

        return sockets;
    }
}
