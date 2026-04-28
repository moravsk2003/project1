package com.rustbuilder.ai.rl.multidiscrete;

/**
 * Immutable representation of a 5-phase multi-discrete action for the RL Agent.
 * 
 * PHASES:
 * 1. typeIndex     (0..11)
 * 2. floorIndex    (0..7)
 * 3. tileIndex     (0..63)
 * 4. rotationIndex (0..3) - 90 degree steps
 * 5. aimSector     (0..24) - 5x5 grid center=12
 */
public class MultiDiscreteAction {
    private final int typeIndex;
    private final int floorIndex;
    private final int tileIndex;
    private final int rotationIndex;
    private final int aimSector;

    public MultiDiscreteAction(int typeIndex, int floorIndex, int tileIndex, int rotationIndex, int aimSector) {
        this.typeIndex = typeIndex;
        this.floorIndex = floorIndex;
        this.tileIndex = tileIndex;
        this.rotationIndex = rotationIndex;
        this.aimSector = aimSector;
    }

    public int getTypeIndex() { return typeIndex; }
    public int getFloorIndex() { return floorIndex; }
    public int getTileIndex() { return tileIndex; }
    public int getTileX() { return tileIndex / MultiDiscreteActionSpace.GRID_SIZE; }
    public int getTileY() { return tileIndex % MultiDiscreteActionSpace.GRID_SIZE; }
    public int getRotationIndex() { return rotationIndex; }
    public int getAimSector() { return aimSector; }

    /**
     * Basic bounds validation for the multi-discrete contract.
     */
    public boolean isValid() {
        return typeIndex >= 0 && typeIndex < MultiDiscreteActionSpace.TYPE_COUNT &&
               floorIndex >= 0 && floorIndex < MultiDiscreteActionSpace.FLOOR_COUNT &&
               tileIndex >= 0 && tileIndex < MultiDiscreteActionSpace.TILE_COUNT &&
               rotationIndex >= 0 && rotationIndex < MultiDiscreteActionSpace.ROTATION_COUNT &&
               aimSector >= 0 && aimSector < MultiDiscreteActionSpace.AIM_SECTOR_COUNT;
    }

    @Override
    public String toString() {
        return String.format("MultiDiscreteAction[type=%d, floor=%d, pos=(%d,%d), rot=%d, aimSector=%d]",
                typeIndex, floorIndex, getTileX(), getTileY(), rotationIndex, aimSector);
    }
}
