package com.rustbuilder.model.deployable;

import com.rustbuilder.model.core.*;
import java.util.Collections;
import java.util.List;

public class ToolCupboard extends BuildingBlock {

    public ToolCupboard(double x, double y, int z, double rotation) {
        super(BuildingType.TC, x, y, z);
        setRotation(rotation);
        // TC takes 1000 wood to build usually? Or depends.
        // Let's set a fixed cost.
    }
    
    @Override
    protected void updateCost() {
        java.util.Map<ResourceType, Integer> cost = new java.util.HashMap<>();
        cost.put(ResourceType.WOOD, 1000); // Standard TC cost
        this.buildCost = java.util.Collections.unmodifiableMap(cost);
    }

    @Override
    public double getUpkeepCost() {
        return 0; // Handled by manager
    }

    @Override
    public double getHealth() {
        return 100;
    }

    @Override
    protected BuildingBlock copyBlock() {
        return new ToolCupboard(getX(), getY(), getZ(), getRotation());
    }

    @Override
    protected List<Socket> computeSockets() {
        return Collections.emptyList(); // TC doesn't have sockets for others to snap TO usually
    }

    @Override
    protected double[] computePolygonPoints() {
        // Small box
        double size = com.rustbuilder.config.GameConstants.TILE_SIZE;
        double cx = getX() + size / 2;
        double cy = getY() + size / 2;
        
        double w = size * 0.18;
        double h = size * 0.30;
        
        // Simple rect
        return new double[] {
            cx - w/2, cy - h/2,
            cx + w/2, cy - h/2,
            cx + w/2, cy + h/2,
            cx - w/2, cy + h/2
        };
    }
}
