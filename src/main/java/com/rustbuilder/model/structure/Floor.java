package com.rustbuilder.model.structure;

import com.rustbuilder.model.core.*;
import java.util.List;

public class Floor extends BuildingBlock {

    public Floor(double x, double y, int z, double rotation) {
        super(BuildingType.FLOOR, x, y, z);
        setRotation(rotation);
    }

    // Removed override getUpkeepCost and getHealth

    @Override
    protected BuildingBlock copyBlock() {
        return new Floor(getX(), getY(), getZ(), getRotation());
    }

    @Override
    protected double[] computePolygonPoints() {
        double size = com.rustbuilder.config.GameConstants.TILE_SIZE;
        double cx = getX() + size / 2;
        double cy = getY() + size / 2;
        double rotRad = Math.toRadians(getRotation());
        double cos = Math.cos(rotRad);
        double sin = Math.sin(rotRad);

        double[][] corners = {
                { -com.rustbuilder.config.GameConstants.HALF_TILE, -com.rustbuilder.config.GameConstants.HALF_TILE },
                { com.rustbuilder.config.GameConstants.HALF_TILE, -com.rustbuilder.config.GameConstants.HALF_TILE },
                { com.rustbuilder.config.GameConstants.HALF_TILE, com.rustbuilder.config.GameConstants.HALF_TILE },
                { -com.rustbuilder.config.GameConstants.HALF_TILE, com.rustbuilder.config.GameConstants.HALF_TILE }
        };

        double[] points = new double[8];
        double scale = 0.98;
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
