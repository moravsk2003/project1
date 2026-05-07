package com.rustbuilder.model.deployable;

import com.rustbuilder.model.core.*;
import java.util.Collections;
import java.util.List;

public class Workbench extends BuildingBlock {

    public Workbench(double x, double y, int z, double rotation) {
        super(BuildingType.WORKBENCH, x, y, z);
        setRotation(rotation);
    }

    @Override
    protected void updateCost() {
        java.util.Map<ResourceType, Integer> cost = new java.util.HashMap<>();
        cost.put(ResourceType.WOOD, 2500);
        cost.put(ResourceType.METAL, 500);
        this.buildCost = java.util.Collections.unmodifiableMap(cost);
    }

    @Override
    public double getUpkeepCost() {
        return 0;
    }

    @Override
    public double getHealth() {
        return 500;
    }

    @Override
    protected BuildingBlock copyBlock() {
        return new Workbench(getX(), getY(), getZ(), getRotation());
    }

    @Override
    protected List<Socket> computeSockets() {
        return Collections.emptyList();
    }

    @Override
    protected double[] computePolygonPoints() {
        double size = com.rustbuilder.config.GameConstants.TILE_SIZE;
        double cx = getX() + size / 2;
        double cy = getY() + size / 2;

        double w = size * 0.5;
        double h = size * 0.35;

        return new double[] {
            cx - w / 2, cy - h / 2,
            cx + w / 2, cy - h / 2,
            cx + w / 2, cy + h / 2,
            cx - w / 2, cy + h / 2
        };
    }
}
