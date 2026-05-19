package com.rustbuilder.ai.rl.environment.state;


import com.rustbuilder.ai.rl.environment.spec.StateEncodingSpec;
import com.rustbuilder.ai.rl.environment.spec.EncodingRuntimeConfig;
import com.rustbuilder.ai.rl.environment.spec.GlobalFeatureSpec;
import com.rustbuilder.model.GridModel;
import java.util.Objects;
import org.nd4j.linalg.api.ndarray.INDArray;

public class HybridV3StateEncoder implements StateRepresentationEncoder {

    private final BucketedVoxelV2StateEncoder voxelEncoder;
    private final GlobalFeatureEncoder globalEncoder;
    private final EncodingRuntimeConfig config;
    
    public HybridV3StateEncoder(EncodingRuntimeConfig config) {
        this.config = config;
        this.voxelEncoder = new BucketedVoxelV2StateEncoder(config);
        this.globalEncoder = new GlobalFeatureEncoder(new GlobalFeatureSpec(config.stateEncodingSpec.globalFeatureCount));
    }

    public HybridV3StateEncoder(EncodingRuntimeConfig config,
                                BucketedVoxelV2StateEncoder voxelEncoder,
                                GlobalFeatureEncoder globalEncoder) {
        this.config = Objects.requireNonNull(config, "config");
        this.voxelEncoder = Objects.requireNonNull(voxelEncoder, "voxelEncoder");
        this.globalEncoder = Objects.requireNonNull(globalEncoder, "globalEncoder");
    }

    @Override
    public EncodedState encode(GridModel gridModel, int selectedTypeOrdinal) {
        long startTime = System.currentTimeMillis();
        
        EncodedState voxelState = voxelEncoder.encode(gridModel, selectedTypeOrdinal);
        INDArray voxelTensor = voxelState.voxelTensor();
        StateEncodingDiagnostics voxelDiag = voxelState.getDiagnostics();
        VoxelAggregationBuffer buffer = voxelEncoder.getBuffer();
        
        GlobalFeatureDiagnostics globalDiag = new GlobalFeatureDiagnostics();
        INDArray globalVector = globalEncoder.encode(gridModel, buffer, voxelDiag, globalDiag);
        
        voxelDiag.encoderTimeMs = System.currentTimeMillis() - startTime;
        voxelDiag.encoderVersion = "v3";
        voxelDiag.globalDiagnostics = globalDiag;
        
        return new EncodedState(voxelTensor, globalVector, null, null, config.stateEncodingSpec, voxelDiag);
    }
}
