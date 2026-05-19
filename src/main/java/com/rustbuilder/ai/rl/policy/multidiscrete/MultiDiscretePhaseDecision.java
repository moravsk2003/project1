package com.rustbuilder.ai.rl.policy.multidiscrete;

/**
 * [RL REDESIGN]
 * Encapsulates the specific chosen indices for each of the 5 multi-discrete phases.
 * This acts as the raw decision vector produced by a controller (heuristic or learned).
 */
public class MultiDiscretePhaseDecision {
    private final int typeIndex;
    private final int floorIndex;
    private final int tileIndex;
    private final int rotationIndex;
    private final int aimSector;

    public MultiDiscretePhaseDecision(int typeIndex, int floorIndex, int tileIndex, 
                                      int rotationIndex, int aimSector) {
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
}
