package com.rustbuilder.model.structure;

import com.rustbuilder.model.core.*;
import java.util.List;

public class TriangleFoundation extends BuildingBlock {

    public TriangleFoundation(double x, double y, int z, double rotation) {
        super(BuildingType.TRIANGLE_FOUNDATION, x, y, z);
        setRotation(rotation);
    }

    // Removed override getUpkeepCost and getHealth

    @Override
    protected double[] computePolygonPoints() {
        double size = com.rustbuilder.config.GameConstants.TILE_SIZE;
        double cx = getX() + size / 2;
        double cy = getY() + size / 2;
        double rotRad = Math.toRadians(getRotation());
        double cos = Math.cos(rotRad);
        double sin = Math.sin(rotRad);

        // Unrotated corners relative to center
        // 1. Bottom Left: -30, 30
        // 2. Top Center: 0, -22 (approx)
        // 3. Bottom Right: 30, 30
        // Note: Top Y is 8. Center Y is 30. 8-30 = -22.
        // Height = 52.

        double[][] corners = {
                { -com.rustbuilder.config.GameConstants.HALF_TILE, com.rustbuilder.config.GameConstants.HALF_TILE },
                { 0, com.rustbuilder.config.GameConstants.TRIANGLE_OFFSET - com.rustbuilder.config.GameConstants.HALF_TILE },
                { com.rustbuilder.config.GameConstants.HALF_TILE, com.rustbuilder.config.GameConstants.HALF_TILE }
        };

        double[] points = new double[6];
        // Shrink factor removed for accurate collision
        double scale = 1.0;
        for (int i = 0; i < 3; i++) {
            double ox = corners[i][0] * scale;
            double oy = corners[i][1] * scale;
            points[i * 2] = cx + (ox * cos - oy * sin);
            points[i * 2 + 1] = cy + (ox * sin + oy * cos);
        }
        return points;
    }

    @Override
    protected List<Socket> computeSockets() {
        return com.rustbuilder.util.SocketGeometryUtils.getTriangleSockets(getX(), getY(), getRotation());
    }
}
