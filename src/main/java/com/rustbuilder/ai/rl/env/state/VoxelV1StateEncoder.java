package com.rustbuilder.ai.rl.env.state;

import com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType;
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
    
    private static final double START_X = 200;
    private static final double START_Y = 200;
    
    public VoxelV1StateEncoder(EncodingRuntimeConfig config) {
        this.config = config;
    }

    @Override
    public EncodedState encode(GridModel gridModel, int selectedTypeOrdinal) {
        int channels = config.stateEncodingSpec.voxelChannels;
        int floors = config.gridSpec.floors;
        int height = config.gridSpec.height;
        int width = config.gridSpec.width;
        
        INDArray tensor = Nd4j.zeros(1, channels, floors, height, width);

        List<BuildingBlock> blocks = gridModel.getAllBlocks();
        for (BuildingBlock b : blocks) {
            int gx = (int) Math.round((b.getX() - START_X) / GameConstants.TILE_SIZE);
            int gy = (int) Math.round((b.getY() - START_Y) / GameConstants.TILE_SIZE);
            int gz = b.getZ();
            
            gx = Math.max(0, Math.min(width - 1, gx));
            gy = Math.max(0, Math.min(height - 1, gy));
            gz = Math.max(0, Math.min(floors - 1, gz));
            
            int channel = mapTypeToChannel(b.getType());
            if (channel >= 0 && channel < 10) {
                tensor.putScalar(new int[]{0, channel, gz, gy, gx}, 1.0f);
            }
        }

        if (selectedTypeOrdinal >= 0 && selectedTypeOrdinal <= 9) {
            double phaseValue = encodePhaseValue(1, 2, selectedTypeOrdinal + 1, 11);
            fillChannel(tensor, 10, phaseValue, floors, height, width);
        }

        return new EncodedState(tensor, config.stateEncodingSpec);
    }

    public static double encodePhaseValue(int phaseIndex, int totalPhases, int value, int maxValue) {
        if (totalPhases <= 1) return 0.0;
        return (double) value / (double) maxValue;
    }

    private static void fillChannel(INDArray tensor, int channel, double value, int floors, int height, int width) {
        for (int z = 0; z < floors; z++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
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
