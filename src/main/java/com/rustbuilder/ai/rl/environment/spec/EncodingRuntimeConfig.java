package com.rustbuilder.ai.rl.environment.spec;

import com.rustbuilder.ai.rl.policy.multidiscrete.MultiDiscreteActionSpace;

public class EncodingRuntimeConfig {
    public final GridSpec gridSpec;
    public final ActionSpaceSpec actionSpaceSpec;
    public final StateEncodingSpec stateEncodingSpec;
    public final CoordinateEncodingMode coordinateEncodingMode;

    public EncodingRuntimeConfig(GridSpec gridSpec, ActionSpaceSpec actionSpaceSpec, StateEncodingSpec stateEncodingSpec, CoordinateEncodingMode coordinateEncodingMode) {
        this.gridSpec = gridSpec;
        this.actionSpaceSpec = actionSpaceSpec;
        this.stateEncodingSpec = stateEncodingSpec;
        this.coordinateEncodingMode = coordinateEncodingMode;
    }

    public static EncodingRuntimeConfig createVoxelV1Config() {
        return new EncodingRuntimeConfig(
                new GridSpec(8, 8, 8, 2048),
                new ActionSpaceSpec(11, 8, 64, MultiDiscreteActionSpace.ROTATION_COUNT, MultiDiscreteActionSpace.AIM_SECTOR_COUNT, TileIndexingMode.LEGACY_64),
                new StateEncodingSpec("Voxel", "v1", 11, new int[]{8, 8, 8}, false, 0, false, false),
                CoordinateEncodingMode.CLAMP
        );
    }
    
    public static EncodingRuntimeConfig createVoxelV2Config() {
        return new EncodingRuntimeConfig(
                new GridSpec(8, 8, 8, 2048),
                new ActionSpaceSpec(11, 8, 64, MultiDiscreteActionSpace.ROTATION_COUNT, MultiDiscreteActionSpace.AIM_SECTOR_COUNT, TileIndexingMode.LEGACY_64),
                new StateEncodingSpec("Voxel", "v2", 16, new int[]{8, 8, 8}, false, 0, false, false),
                CoordinateEncodingMode.SKIP_OUT_OF_BOUNDS
        );
    }
    
    public static EncodingRuntimeConfig createHybridV3Config() {
        return new EncodingRuntimeConfig(
                new GridSpec(8, 8, 8, 2048),
                new ActionSpaceSpec(11, 8, 64, MultiDiscreteActionSpace.ROTATION_COUNT, MultiDiscreteActionSpace.AIM_SECTOR_COUNT, TileIndexingMode.LEGACY_64),
                new StateEncodingSpec("HYBRID_V3_VOXEL_GLOBAL", "v3", 16, new int[]{8, 8, 8}, true, 32, false, false),
                CoordinateEncodingMode.SKIP_OUT_OF_BOUNDS
        );
    }
}
