package com.rustbuilder.ai.rl.env.state;

import com.rustbuilder.core.action.BuildAction.ActionType;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.config.GameConstants;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import com.rustbuilder.ai.rl.env.spec.EncodingRuntimeConfig;
import java.util.List;

public class VoxelV1StateEncoder implements StateRepresentationEncoder {
    
    private final EncodingRuntimeConfig config;
    
    public VoxelV1StateEncoder(EncodingRuntimeConfig config) {
        this.config = config;
    }

    @Override
    public EncodedState encode(GridModel gridModel, int selectedTypeOrdinal) {
        int channels = config.stateEncodingSpec.voxelChannels;
        int floors = config.gridSpec.floors;
        int height = config.gridSpec.height;
        int width = config.gridSpec.width;
        
        float[] flatTensor = new float[channels * floors * height * width];

        List<BuildingBlock> blocks = gridModel.getAllBlocks();
        for (BuildingBlock b : blocks) {
            int gx = (int) Math.round((b.getX() + GameConstants.HALF_TILE - GameConstants.GRID_ORIGIN_X) / GameConstants.TILE_SIZE);
            int gy = (int) Math.round((b.getY() + GameConstants.HALF_TILE - GameConstants.GRID_ORIGIN_Y) / GameConstants.TILE_SIZE);
            int gz = b.getZ();
            
            gx = Math.max(0, Math.min(width - 1, gx));
            gy = Math.max(0, Math.min(height - 1, gy));
            gz = Math.max(0, Math.min(floors - 1, gz));
            
            int channel = mapTypeToChannel(b.getType());
            if (channel >= 0 && channel < 10) {
                flatTensor[tensorIndex(channel, gz, gy, gx, floors, height, width)] = 1.0f;
            }
        }

        if (selectedTypeOrdinal >= 0 && selectedTypeOrdinal <= 9) {
            double phaseValue = encodePhaseValue(1, 2, selectedTypeOrdinal + 1, 11);
            fillChannel(flatTensor, 10, (float) phaseValue, floors, height, width);
        }

        INDArray tensor = Nd4j.create(flatTensor, new int[]{1, channels, floors, height, width}, 'c');
        return new EncodedState(tensor, config.stateEncodingSpec);
    }

    public static double encodePhaseValue(int phaseIndex, int totalPhases, int value, int maxValue) {
        if (totalPhases <= 1) return 0.0;
        return (double) value / (double) maxValue;
    }

    private static void fillChannel(float[] flatTensor, int channel, float value, int floors, int height, int width) {
        for (int z = 0; z < floors; z++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    flatTensor[tensorIndex(channel, z, y, x, floors, height, width)] = value;
                }
            }
        }
    }

    private static int tensorIndex(int c, int z, int y, int x, int floors, int height, int width) {
        return (((c * floors) + z) * height + y) * width + x;
    }

    private static int mapTypeToChannel(BuildingType type) {
        switch (type) {
            case FOUNDATION: return ActionType.FOUNDATION.ordinal();
            case TRIANGLE_FOUNDATION: return ActionType.TRIANGLE_FOUNDATION.ordinal();
            case WALL: return ActionType.WALL.ordinal();
            case DOORWAY: return ActionType.DOORWAY.ordinal();
            case WINDOW_FRAME: return ActionType.WINDOW_FRAME.ordinal();
            case DOOR: return ActionType.DOORWAY.ordinal(); 
            case FLOOR: return ActionType.FLOOR.ordinal();
            case TRIANGLE_FLOOR: return ActionType.TRIANGLE_FLOOR.ordinal();
            case TC: return ActionType.TC.ordinal();
            case WORKBENCH: return ActionType.WORKBENCH.ordinal();
            case LOOT_ROOM: return ActionType.LOOT_ROOM.ordinal();
            default: return -1;
        }
    }
}
