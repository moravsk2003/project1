package com.rustbuilder.model.structure;

import com.rustbuilder.model.core.*;
import java.util.List;

public class Foundation extends BuildingBlock {

    public Foundation(double x, double y, int z) {
        this(x, y, z, 0);
    }

    public Foundation(double x, double y, int z, double rotation) {
        super(BuildingType.FOUNDATION, x, y, z);
        setRotation(rotation);
    }

    // ... (upkeep and health unchanged)
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
        // TL: -30, -30
        // TR: 30, -30
        // BR: 30, 30
        // BL: -30, 30
        double[][] corners = {
                { -com.rustbuilder.config.GameConstants.HALF_TILE, -com.rustbuilder.config.GameConstants.HALF_TILE },
                { com.rustbuilder.config.GameConstants.HALF_TILE, -com.rustbuilder.config.GameConstants.HALF_TILE },
                { com.rustbuilder.config.GameConstants.HALF_TILE, com.rustbuilder.config.GameConstants.HALF_TILE },
                { -com.rustbuilder.config.GameConstants.HALF_TILE, com.rustbuilder.config.GameConstants.HALF_TILE }
        };

        double[] points = new double[8];
        // Shrink factor to prevent false positives on touching blocks
        double scale = 0.95;
        for (int i = 0; i < 4; i++) {
            double ox = corners[i][0] * scale;
            double oy = corners[i][1] * scale;
            points[i * 2] = cx + (ox * cos - oy * sin);
            points[i * 2 + 1] = cy + (ox * sin + oy * cos);
        }
        return points;
    }

    @Override
    protected List<Socket> computeSockets() {
        return com.rustbuilder.util.SocketGeometryUtils.getSquareSockets(getX(), getY(), getRotation());
    }
}
