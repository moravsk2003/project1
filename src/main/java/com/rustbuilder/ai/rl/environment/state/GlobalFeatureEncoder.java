package com.rustbuilder.ai.rl.environment.state;

import com.rustbuilder.ai.rl.environment.spec.GlobalFeatureSpec;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.List;

public class GlobalFeatureEncoder {
    
    private final GlobalFeatureSpec spec;
    
    public GlobalFeatureEncoder(GlobalFeatureSpec spec) {
        this.spec = spec;
    }
    
    public INDArray encode(GridModel grid, VoxelAggregationBuffer buffer, StateEncodingDiagnostics voxelDiag, GlobalFeatureDiagnostics outDiag) {
        int N = spec.featureCount;
        INDArray vector = Nd4j.zeros(1, N);
        
        List<BuildingBlock> blocks = grid.getAllBlocks();
        int blockCount = blocks.size();
        
        float blockCountNorm = Math.min(blockCount / 150f, 1.0f);
        float occupiedVoxelCountNorm = Math.min(voxelDiag.occupiedVoxelsCount / 200f, 1.0f);
        float multiBlockVoxelRatio = voxelDiag.occupiedVoxelsCount > 0 ? (float) voxelDiag.multiBlockVoxelCount / voxelDiag.occupiedVoxelsCount : 0f;
        float sameTypeMultiBlockVoxelRatio = voxelDiag.occupiedVoxelsCount > 0 ? (float) voxelDiag.sameTypeMultiBlockVoxelCount / voxelDiag.occupiedVoxelsCount : 0f;
        float maxBlocksPerVoxelNorm = Math.min(voxelDiag.maxBlocksPerVoxel / 10f, 1.0f);
        float avgBlocksPerOccupiedVoxelNorm = Math.min(voxelDiag.avgBlocksPerOccupiedVoxel / 5f, 1.0f);
        
        boolean hasTc = false;
        int maxFloor = 0;
        int usedFloorsMask = 0;
        float sumStability = 0f;
        float minStability = 1.0f;
        int structuralCount = 0;
        int lowStabilityCount = 0;
        
        int foundationCount = 0;
        int wallCount = 0;
        int floorCount = 0;
        int deployableCount = 0;
        
        for (BuildingBlock b : blocks) {
            BuildingType t = b.getType();
            if (t == BuildingType.TC) hasTc = true;
            
            int f = b.getZ();
            if (f > maxFloor) maxFloor = f;
            if (f >= 0 && f < 32) usedFloorsMask |= (1 << f);
            
            if (isStructural(t)) {
                structuralCount++;
                float stab = (float) Math.max(0.0, Math.min(1.0, b.getStability()));
                sumStability += stab;
                if (stab < minStability) minStability = stab;
                if (stab < 0.3f) lowStabilityCount++;
            }
            
            if (t == BuildingType.FOUNDATION || t == BuildingType.TRIANGLE_FOUNDATION) foundationCount++;
            else if (t == BuildingType.WALL || t == BuildingType.DOORWAY || t == BuildingType.WINDOW_FRAME) wallCount++;
            else if (t == BuildingType.FLOOR || t == BuildingType.TRIANGLE_FLOOR) floorCount++;
            else deployableCount++;
        }
        
        float avgStructuralStability = structuralCount > 0 ? sumStability / structuralCount : 0f;
        if (structuralCount == 0) minStability = 0f;
        float lowStabilityBlockRatio = structuralCount > 0 ? (float) lowStabilityCount / structuralCount : 0f;
        
        float tcPresent = hasTc ? 1.0f : 0.0f;
        float tcCountNorm = hasTc ? 1.0f : 0.0f; 
        float maxFloorNorm = Math.min(maxFloor / 8f, 1.0f);
        float usedFloorCountNorm = Math.min(Integer.bitCount(usedFloorsMask) / 8f, 1.0f);
        
        float foundationRatio = blockCount > 0 ? (float) foundationCount / blockCount : 0f;
        float wallRatio = blockCount > 0 ? (float) wallCount / blockCount : 0f;
        float floorRatio = blockCount > 0 ? (float) floorCount / blockCount : 0f;
        float deployableRatio = blockCount > 0 ? (float) deployableCount / blockCount : 0f;
        
        float hasAnyStructure = structuralCount > 0 ? 1.0f : 0.0f;
        float hasMultiFloorStructure = maxFloor > 0 ? 1.0f : 0.0f;
        
        vector.putScalar(0, 0, blockCountNorm);
        vector.putScalar(0, 1, occupiedVoxelCountNorm);
        vector.putScalar(0, 2, multiBlockVoxelRatio);
        vector.putScalar(0, 3, sameTypeMultiBlockVoxelRatio);
        vector.putScalar(0, 4, maxBlocksPerVoxelNorm);
        vector.putScalar(0, 5, avgBlocksPerOccupiedVoxelNorm);
        vector.putScalar(0, 6, tcPresent);
        vector.putScalar(0, 7, tcCountNorm);
        vector.putScalar(0, 8, maxFloorNorm);
        vector.putScalar(0, 9, usedFloorCountNorm);
        vector.putScalar(0, 10, avgStructuralStability);
        vector.putScalar(0, 11, minStability);
        vector.putScalar(0, 12, lowStabilityBlockRatio);
        vector.putScalar(0, 13, voxelDiag.outOfBoundsRatio);
        vector.putScalar(0, 14, voxelDiag.encodedBlocksCount > 0 ? 1.0f : 0.0f);
        vector.putScalar(0, 15, foundationRatio);
        vector.putScalar(0, 16, wallRatio);
        vector.putScalar(0, 17, floorRatio);
        vector.putScalar(0, 18, deployableRatio);
        vector.putScalar(0, 19, hasAnyStructure);
        vector.putScalar(0, 20, hasMultiFloorStructure);
        
        if (outDiag != null) {
            outDiag.globalFeatureCount = N;
            int nonzero = 0;
            float gMin = Float.MAX_VALUE;
            float gMax = -Float.MAX_VALUE;
            float gSum = 0;
            for (int i = 0; i < N; i++) {
                float v = vector.getFloat(0, i);
                if (v != 0f) nonzero++;
                if (v < gMin) gMin = v;
                if (v > gMax) gMax = v;
                gSum += v;
            }
            outDiag.globalNonzeroCount = nonzero;
            outDiag.globalMin = gMin == Float.MAX_VALUE ? 0f : gMin;
            outDiag.globalMax = gMax == -Float.MAX_VALUE ? 0f : gMax;
            outDiag.globalMean = gSum / N;
            
            outDiag.blockCountNorm = blockCountNorm;
            outDiag.occupiedVoxelCountNorm = occupiedVoxelCountNorm;
            outDiag.multiBlockVoxelRatio = multiBlockVoxelRatio;
            outDiag.tcPresent = tcPresent;
            outDiag.maxFloorNorm = maxFloorNorm;
            outDiag.avgStructuralStability = avgStructuralStability;
            outDiag.outOfBoundsRatio = voxelDiag.outOfBoundsRatio;
        }
        
        return vector;
    }
    
    private boolean isStructural(BuildingType type) {
        switch (type) {
            case FOUNDATION: case TRIANGLE_FOUNDATION:
            case WALL: case DOORWAY: case WINDOW_FRAME:
            case FLOOR: case TRIANGLE_FLOOR:
                return true;
            default: return false;
        }
    }
}
