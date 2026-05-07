package com.rustbuilder.model.deployable;

import com.rustbuilder.model.core.*;
import java.util.Collections;
import java.util.List;

public class LootRoom extends BuildingBlock {

    public LootRoom(double x, double y, int z, double rotation) {
        super(BuildingType.LOOT_ROOM, x, y, z);
        setRotation(rotation);
    }

    @Override
    protected void updateCost() {
        java.util.Map<ResourceType, Integer> cost = new java.util.HashMap<>();
        cost.put(ResourceType.WOOD, 250);
        cost.put(ResourceType.METAL, 50);
        this.buildCost = java.util.Collections.unmodifiableMap(cost);
    }

    @Override
    public double getUpkeepCost() {
        return 0;
    }

    @Override
    public double getHealth() {
        return 200;
    }

    @Override
    protected BuildingBlock copyBlock() {
        return new LootRoom(getX(), getY(), getZ(), getRotation());
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

        double w = size * 0.4;
        double h = size * 0.3;

        return new double[] {
            cx - w / 2, cy - h / 2,
            cx + w / 2, cy - h / 2,
            cx + w / 2, cy + h / 2,
            cx - w / 2, cy + h / 2
        };
    }
}
