package com.rustbuilder.ai.rl;

import com.rustbuilder.ai.rl.env.spec.EncodingRuntimeConfig;
import com.rustbuilder.ai.rl.env.state.BucketedVoxelV2StateEncoder;
import com.rustbuilder.ai.rl.env.state.HybridV3StateEncoder;
import com.rustbuilder.ai.rl.env.state.VoxelV1StateEncoder;

public final class DefaultStateEncoderFactory implements StateEncoderFactory {

    @Override
    public StateEncoderBundle create(RLTrainingService.EncoderMode mode) {
        RLTrainingService.EncoderMode selectedMode =
                mode != null ? mode : RLTrainingService.EncoderMode.V3;
        switch (selectedMode) {
            case V1: {
                EncodingRuntimeConfig config = EncodingRuntimeConfig.createVoxelV1Config();
                return new StateEncoderBundle(config, new VoxelV1StateEncoder(config));
            }
            case V2: {
                EncodingRuntimeConfig config = EncodingRuntimeConfig.createVoxelV2Config();
                return new StateEncoderBundle(config, new BucketedVoxelV2StateEncoder(config));
            }
            case V3:
            default: {
                EncodingRuntimeConfig config = EncodingRuntimeConfig.createHybridV3Config();
                return new StateEncoderBundle(config, new HybridV3StateEncoder(config));
            }
        }
    }
}
