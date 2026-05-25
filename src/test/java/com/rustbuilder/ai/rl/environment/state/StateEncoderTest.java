package com.rustbuilder.ai.rl.environment.state;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import com.rustbuilder.ai.rl.environment.spec.EncodingRuntimeConfig;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.structure.Foundation;
import com.rustbuilder.model.structure.Wall;
import com.rustbuilder.model.core.Orientation;

public class StateEncoderTest {

    private GridModel gridModel;

    @BeforeEach
    void setUp() {
        gridModel = new GridModel();
    }

    @Test
    void testVoxelV1StateEncoderShape() {
        EncodingRuntimeConfig config = EncodingRuntimeConfig.createVoxelV1Config();
        VoxelV1StateEncoder encoder = new VoxelV1StateEncoder(config);

        // Place one foundation
        Foundation foundation = new Foundation(0, 0, 0);
        gridModel.addBlock(foundation);

        try (EncodedState state = encoder.encode(gridModel, 0)) {
            assertNotNull(state);
            INDArray voxelTensor = state.voxelTensor();
            assertNotNull(voxelTensor);
            
            // Expected shape: [1, 11, 8, 8, 8]
            long[] shape = voxelTensor.shape();
            assertEquals(5, shape.length);
            assertEquals(1, shape[0]);
            assertEquals(11, shape[1]);
            assertEquals(8, shape[2]);
            assertEquals(8, shape[3]);
            assertEquals(8, shape[4]);
            
            assertNull(state.getGlobalVector());
        }
    }

    @Test
    void testBucketedVoxelV2StateEncoderShape() {
        EncodingRuntimeConfig config = EncodingRuntimeConfig.createVoxelV2Config();
        BucketedVoxelV2StateEncoder encoder = new BucketedVoxelV2StateEncoder(config);

        Foundation foundation = new Foundation(0, 0, 0);
        gridModel.addBlock(foundation);

        try (EncodedState state = encoder.encode(gridModel, 0)) {
            assertNotNull(state);
            INDArray voxelTensor = state.voxelTensor();
            assertNotNull(voxelTensor);

            // Expected shape: [1, 16, 8, 8, 8]
            long[] shape = voxelTensor.shape();
            assertEquals(5, shape.length);
            assertEquals(1, shape[0]);
            assertEquals(16, shape[1]);
            assertEquals(8, shape[2]);
            assertEquals(8, shape[3]);
            assertEquals(8, shape[4]);

            assertNull(state.getGlobalVector());
        }
    }

    @Test
    void testHybridV3StateEncoderShapeAndGlobalVector() {
        EncodingRuntimeConfig config = EncodingRuntimeConfig.createHybridV3Config();
        HybridV3StateEncoder encoder = new HybridV3StateEncoder(config);

        Foundation foundation = new Foundation(0, 0, 0);
        gridModel.addBlock(foundation);

        try (EncodedState state = encoder.encode(gridModel, 0)) {
            assertNotNull(state);
            INDArray voxelTensor = state.voxelTensor();
            assertNotNull(voxelTensor);

            // Expected shape: [1, 16, 8, 8, 8]
            long[] shape = voxelTensor.shape();
            assertEquals(5, shape.length);
            assertEquals(1, shape[0]);
            assertEquals(16, shape[1]);
            assertEquals(8, shape[2]);
            assertEquals(8, shape[3]);
            assertEquals(8, shape[4]);

            INDArray globalVector = state.getGlobalVector();
            assertNotNull(globalVector);
            long[] globalShape = globalVector.shape();
            assertEquals(2, globalShape.length);
            assertEquals(1, globalShape[0]);
            assertEquals(32, globalShape[1]);
        }
    }
}
