package com.rustbuilder.ai.rl.multidiscrete;

import com.rustbuilder.ai.rl.legacy.*;
import com.rustbuilder.ai.ea.BaseGenome.BuildAction;

/**
 * [RL REDESIGN]
 * Mapper/Adapter to convert between the new 5-phase MultiDiscreteAction 
 * and the legacy 2-phase BuildAction system.
 * 
 * This class serves as a "bridge" to allow the RL agent to use the new action 
 * representation while still being compatible with the established placement 
 * and geometry layers.
 */
public class MultiDiscreteActionMapper {
    
    /**
     * Maps legacy BuildAction to RL MultiDiscreteAction.
     */
    public static MultiDiscreteAction toMultiDiscrete(BuildAction legacyAction) {
        if (legacyAction == null) return null;

        int typeIdx = ActionSpace.encodeType(legacyAction.actionType);
        int floorIdx = legacyAction.floor;
        
        // Combine grid coordinates into a singular tileIndex (0..63)
        int tileIndex = legacyAction.gridX * MultiDiscreteActionSpace.GRID_SIZE + legacyAction.gridY;

        // Map rotation index
        int rotIdx = fromLegacyRotation(legacyAction.orientation);

        // AimSector: use the one from legacyAction
        int aimSector = legacyAction.aimSector;

        return new MultiDiscreteAction(typeIdx, floorIdx, tileIndex, rotIdx, aimSector);
    }

    /**
     * Bridge utility to convert between RL MultiDiscreteAction (5 phases)
     * and legacy RustBuilder BuildAction.
     * 
     * MAPPING STRATEGY (Legacy Mode):
     * - typeIndex -> BuildingType (via ActionSpace)
     * - gridX, gridY, floorIndex -> 1:1 mapping
     * - rotationIndex (0..3) -> orientation (0..3): 1:1 mapping.
     * - aimSector -> Currently ignored (placeholder for future tile-local refinement).
     * - tier/doorType -> Default values (STONE/SHEET_METAL).
     * 
     * @param multiAction The multi-discrete action to convert
     * @return A legacy BuildAction compatible with current GridPlacementUtils
     */
    public static BuildAction toBuildAction(MultiDiscreteAction multiAction) {
        if (multiAction == null) return null;

        BuildAction.ActionType type = ActionSpace.decodeType(multiAction.getTypeIndex());
        
        // 1:1 mapping: rotation buckets [0..3] -> cards [0..3]
        int orientation = Math.min(3, Math.max(0, multiAction.getRotationIndex()));

        // Use standard tier (STONE) and doorType (SHEET_METAL) for compatibility
        int defaultTier = 2;
        int defaultDoorType = 0;

        return new BuildAction(
            type,
            multiAction.getTileX(),
            multiAction.getTileY(),
            multiAction.getFloorIndex(),
            orientation,
            defaultTier,
            defaultDoorType,
            multiAction.getAimSector()
        );
    }

    /**
     * Helper to map an orientation back to a 4-bucket rotation index.
     * This is the inverse of the logic in toBuildAction.
     * 
     * @param orientation Legacy orientation (0..3)
     * @return Rotation bucket index (0..3)
     */
    public static int fromLegacyRotation(int orientation) {
        return Math.min(3, Math.max(0, orientation));
    }

    /**
     * Decode an aim sector index (0..24) into a 5x5 grid coordinate.
     */
    public static int[] decodeAimSector(int sector) {
        int x = sector % 5;
        int y = sector / 5;
        return new int[]{x, y};
    }
}
