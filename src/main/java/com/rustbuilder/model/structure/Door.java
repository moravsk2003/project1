package com.rustbuilder.model.structure;

import com.rustbuilder.model.core.*;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A door that is placed inside a DOORWAY (door frame).
 * It takes the same position, orientation, and rotation as its parent doorway.
 */
public class Door extends BuildingBlock {

    private Orientation orientation;
    private DoorType doorType;

    public Door(double x, double y, int z, Orientation orientation, DoorType doorType) {
        super(BuildingType.DOOR, x, y, z);
        this.orientation = orientation;
        this.doorType = doorType;
        updateCost();
    }

    public Orientation getOrientation() {
        return orientation;
    }

    public void setOrientation(Orientation orientation) {
        this.orientation = orientation;
    }

    public DoorType getDoorType() {
        return doorType;
    }

    public void setDoorType(DoorType doorType) {
        this.doorType = doorType;
        updateCost();
    }

    @Override
    public double getUpkeepCost() {
        return 0; // Doors don't have upkeep in Rust
    }

    @Override
    public double getHealth() {
        switch (doorType) {
            case SHEET_METAL: return 250;
            case GARAGE:      return 600;
            case ARMORED:     return 800;
            default:          return 250;
        }
    }

    @Override
    protected BuildingBlock copyBlock() {
        return new Door(getX(), getY(), getZ(), orientation, doorType);
    }

    @Override
    protected void updateCost() {
        Map<ResourceType, Integer> cost = new HashMap<>();
        if (doorType == null) return;
        switch (doorType) {
            case SHEET_METAL:
                cost.put(ResourceType.METAL, 150);
                break;
            case GARAGE:
                cost.put(ResourceType.METAL, 300);
                cost.put(ResourceType.WOOD, 75);
                break;
            case ARMORED:
                cost.put(ResourceType.HQM, 25);
                cost.put(ResourceType.METAL, 200);
                break;
        }
        this.buildCost = Collections.unmodifiableMap(cost);
    }

    @Override
    protected double[] computePolygonPoints() {
        // Same shape as a wall but rendered differently (visual only)
        return new double[0]; // No collision polygon — door sits inside doorway
    }

    @Override
    protected double[] computeCollisionPoints() {
        return new double[0]; // No collision — resides inside a doorway
    }

    @Override
    protected List<Socket> computeSockets() {
        return Collections.emptyList(); // Doors don't have sockets
    }
}
