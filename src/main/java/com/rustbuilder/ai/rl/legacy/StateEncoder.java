package com.rustbuilder.ai.rl.legacy;

import com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.config.GameConstants;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.List;

/**
 * Encodes the current GridModel state into an INDArray for a DL4J Convolutional Neural Network.
 * 
 * Shape: [BatchSize=1, Channels=11, Depth(Z)=8, Height(Y)=8, Width(X)=8]
 * 
 * Channels 0-9:  Block type presence (one channel per ActionType)
 * Channel 10:    Phase channel — encodes which block type was selected in Phase 1
 *                  0.0 = Phase 1 (selecting type)
 *                  (typeOrdinal+1)/11.0 = Phase 2 (selecting position for that type)
 */
public class StateEncoder {

    public static final int GRID_SIZE = 8;
    public static final int MAX_FLOORS = 8;
    public static final int CHANNELS = 11; // 10 block types + 1 phase channel
    
    private static final double START_X = 200;
    private static final double START_Y = 200;

    /**
     * Encode state for Phase 1 (type selection). Phase channel = all zeros.
     */
    public static INDArray encode(GridModel gridModel) {
        return encodeWithPhase(gridModel, -1);
    }

    /**
     * Encode state with phase information.
     * @param selectedTypeOrdinal -1 for Phase 1 (selecting type), 0-9 for Phase 2 (selecting position)
     */
    public static INDArray encodeWithPhase(GridModel gridModel, int selectedTypeOrdinal) {
        INDArray tensor = Nd4j.zeros(1, CHANNELS, MAX_FLOORS, GRID_SIZE, GRID_SIZE);

        // Channels 0-9: Block presence
        List<BuildingBlock> blocks = gridModel.getAllBlocks();
        for (BuildingBlock b : blocks) {
            int gx = (int) Math.round((b.getX() - START_X) / GameConstants.TILE_SIZE);
            int gy = (int) Math.round((b.getY() - START_Y) / GameConstants.TILE_SIZE);
            int gz = b.getZ();
            
            gx = Math.max(0, Math.min(GRID_SIZE - 1, gx));
            gy = Math.max(0, Math.min(GRID_SIZE - 1, gy));
            gz = Math.max(0, Math.min(MAX_FLOORS - 1, gz));
            
            int channel = mapTypeToChannel(b.getType());
            if (channel >= 0 && channel < 10) {
                tensor.putScalar(new int[]{0, channel, gz, gy, gx}, 1.0f);
            }
        }

        // Channel 10: Phase encoding
        if (selectedTypeOrdinal >= 0 && selectedTypeOrdinal <= 9) {
            // Phase 2: fill entire channel with a value encoding the selected type
            // PRESERVE: Current 2-phase logic (selected type ordinal + 1) / 11.0
            double phaseValue = encodePhaseValue(1, 2, selectedTypeOrdinal + 1, 11);
            fillChannel(tensor, 10, phaseValue);
        }
        // Phase 1: channel 10 stays all zeros (already initialized)

        return tensor;
    }

    /**
     * Helper to encode a phase value into a normalized double for the CNN.
     * 
     * IMPORTANT: This implementation is currently tailored for the legacy 2-phase 
     * (type selection -> position selection) training loop. 
     * It is NOT used for the new 7-phase multi-discrete action system yet.
     * 
     * @param phaseIndex Current phase (0-based)
     * @param totalPhases Total phases in the action cycle
     * @param value The selected value within the phase (ordinal + 1 for type selection)
     * @param maxValue Max possible value for this phase (e.g., 11 for type selection)
     * @return Normalized value in range [0, 1]
     */
    public static double encodePhaseValue(int phaseIndex, int totalPhases, int value, int maxValue) {
        if (totalPhases <= 1) return 0.0;
        // Current 2-phase logic uses (value + 1) / 11.0, where value is ordinal.
        // We ensure this helper can be expanded for the new 7-phase flow.
        return (double) value / (double) maxValue;
    }

    private static void fillChannel(INDArray tensor, int channel, double value) {
        for (int z = 0; z < MAX_FLOORS; z++) {
            for (int y = 0; y < GRID_SIZE; y++) {
                for (int x = 0; x < GRID_SIZE; x++) {
                    tensor.putScalar(new int[]{0, channel, z, y, x}, (float)value);
                }
            }
        }
    }

    private static int mapTypeToChannel(BuildingType type) {
        switch (type) {
            case FOUNDATION: return ActionType.FOUNDATION.ordinal();
            case TRIANGLE_FOUNDATION: return ActionType.TRIANGLE_FOUNDATION.ordinal();
            case WALL: return ActionType.WALL.ordinal();
            case DOORWAY: return ActionType.DOORWAY.ordinal();
            case WINDOW_FRAME: return ActionType.WINDOW_FRAME.ordinal();
            case DOOR: return ActionType.DOORWAY.ordinal(); // Door is auto-placed with doorway
            case FLOOR: return ActionType.FLOOR.ordinal();
            case TRIANGLE_FLOOR: return ActionType.TRIANGLE_FLOOR.ordinal();
            case TC: return ActionType.TC.ordinal();
            case WORKBENCH: return ActionType.WORKBENCH.ordinal();
            case LOOT_ROOM: return ActionType.LOOT_ROOM.ordinal();
            default: return -1;
        }
    }
}
