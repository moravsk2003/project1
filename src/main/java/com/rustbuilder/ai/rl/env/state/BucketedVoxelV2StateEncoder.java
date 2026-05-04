package com.rustbuilder.ai.rl.env.state;

import com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType;
import com.rustbuilder.ai.rl.env.spec.CoordinateEncodingMode;
import com.rustbuilder.ai.rl.env.spec.EncodingRuntimeConfig;
import com.rustbuilder.config.GameConstants;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.List;

public class BucketedVoxelV2StateEncoder implements StateRepresentationEncoder {

    private final EncodingRuntimeConfig config;
    private final VoxelAggregationBuffer buffer;
    
    private static final double START_X = 200;
    private static final double START_Y = 200;
    
    public BucketedVoxelV2StateEncoder(EncodingRuntimeConfig config) {
        this.config = config;
        this.buffer = new VoxelAggregationBuffer(
            config.gridSpec.width, 
            config.gridSpec.height, 
            config.gridSpec.floors
        );
    }
    
    public VoxelAggregationBuffer getBuffer() {
        return this.buffer;
    }
    
    @Override
    public EncodedState encode(GridModel gridModel, int selectedTypeOrdinal) {
        long startTime = System.currentTimeMillis();
        
        buffer.reset();
        
        StateEncodingDiagnostics diagnostics = new StateEncodingDiagnostics();
        diagnostics.encoderVersion = "v2";
        diagnostics.voxelChannels = 16;
        
        int encodedBlocks = 0;
        int skippedBlocks = 0;
        
        int gridW = config.gridSpec.width;
        int gridH = config.gridSpec.height;
        int gridF = config.gridSpec.floors;
        
        List<BuildingBlock> blocks = gridModel.getAllBlocks();
        
        double tcX = -1, tcY = -1, tcZ = -1;
        boolean hasAnyTc = false;
        
        for (BuildingBlock b : blocks) {
            int gx = (int) Math.round((b.getX() - START_X) / GameConstants.TILE_SIZE);
            int gy = (int) Math.round((b.getY() - START_Y) / GameConstants.TILE_SIZE);
            int gz = b.getZ();
            
            if (config.coordinateEncodingMode == CoordinateEncodingMode.SKIP_OUT_OF_BOUNDS) {
                if (gx < 0 || gx >= gridW || gy < 0 || gy >= gridH || gz < 0 || gz >= gridF) {
                    skippedBlocks++;
                    continue;
                }
            } else {
                gx = Math.max(0, Math.min(gridW - 1, gx));
                gy = Math.max(0, Math.min(gridH - 1, gy));
                gz = Math.max(0, Math.min(gridF - 1, gz));
            }
            
            int idx = buffer.getIndex(gx, gy, gz);
            if (idx == -1) {
                skippedBlocks++;
                continue;
            }
            
            encodedBlocks++;
            buffer.blockCounts[idx]++;
            
            int typeId = mapTypeToChannel(b.getType());
            if (typeId >= 0 && typeId < 10) {
                buffer.typeMasks[idx] |= (1 << typeId);
            }
            
            if (isStructural(b.getType())) {
                float stab = (float) Math.max(0.0, Math.min(1.0, b.getStability()));
                if (stab > buffer.structuralStabilityMax[idx]) {
                    buffer.structuralStabilityMax[idx] = stab;
                }
            }
            
            if (b.getType() == BuildingType.TC) {
                buffer.hasTc[idx] = true;
                tcX = gx; tcY = gy; tcZ = gz;
                hasAnyTc = true;
            }
        }
        
        diagnostics.encodedBlocksCount = encodedBlocks;
        diagnostics.skippedOutOfBoundsBlocks = skippedBlocks;
        diagnostics.outOfBoundsRatio = (encodedBlocks + skippedBlocks > 0) ? 
            (float) skippedBlocks / (encodedBlocks + skippedBlocks) : 0f;
            
        INDArray tensor = Nd4j.zeros(1, 16, gridF, gridH, gridW);
        
        int maxBlocks = 0;
        int occupiedCount = 0;
        int multiBlockCount = 0;
        int sameTypeMultiCount = 0;
        
        double maxDist = Math.sqrt(gridW * gridW + gridH * gridH + gridF * gridF);
        
        for (int z = 0; z < gridF; z++) {
            for (int y = 0; y < gridH; y++) {
                for (int x = 0; x < gridW; x++) {
                    int idx = buffer.getIndex(x, y, z);
                    int count = buffer.blockCounts[idx];
                    
                    if (count > 0) {
                        occupiedCount++;
                        if (count > maxBlocks) maxBlocks = count;
                        if (count > 1) {
                            multiBlockCount++;
                            if (Integer.bitCount(buffer.typeMasks[idx]) == 1) {
                                sameTypeMultiCount++;
                            }
                        }
                        
                        for (int c = 0; c < 10; c++) {
                            if ((buffer.typeMasks[idx] & (1 << c)) != 0) {
                                tensor.putScalar(new int[]{0, c, z, y, x}, 1.0);
                            }
                        }
                        
                        tensor.putScalar(new int[]{0, 10, z, y, x}, 1.0);
                        diagnostics.occupancyBinaryNonzero++;
                        
                        double density = Math.min(count / 4.0, 1.0);
                        tensor.putScalar(new int[]{0, 11, z, y, x}, density);
                        diagnostics.occupancyDensityNonzero++;
                        
                        if (buffer.structuralStabilityMax[idx] > 0) {
                            tensor.putScalar(new int[]{0, 12, z, y, x}, buffer.structuralStabilityMax[idx]);
                            diagnostics.stabilityNonzero++;
                        }
                    }
                    
                    if (z == 0 || (idx >= gridW * gridH && buffer.blockCounts[idx - gridW * gridH] > 0)) {
                        tensor.putScalar(new int[]{0, 13, z, y, x}, 1.0);
                        diagnostics.supportBelowNonzero++;
                    }
                    
                    if (isNearStructure(x, y, z, gridW, gridH, gridF)) {
                        tensor.putScalar(new int[]{0, 14, z, y, x}, 1.0);
                        diagnostics.nearStructureNonzero++;
                    }
                    
                    if (hasAnyTc) {
                        double dist = Math.sqrt(Math.pow(x - tcX, 2) + Math.pow(y - tcY, 2) + Math.pow(z - tcZ, 2));
                        double prox = 1.0 - Math.min(dist / maxDist, 1.0);
                        if (prox > 0) {
                            tensor.putScalar(new int[]{0, 15, z, y, x}, prox);
                            diagnostics.tcProximityNonzero++;
                        }
                    }
                }
            }
        }
        
        diagnostics.occupiedVoxelsCount = occupiedCount;
        diagnostics.multiBlockVoxelCount = multiBlockCount;
        diagnostics.sameTypeMultiBlockVoxelCount = sameTypeMultiCount;
        diagnostics.maxBlocksPerVoxel = maxBlocks;
        diagnostics.avgBlocksPerOccupiedVoxel = occupiedCount > 0 ? (float) encodedBlocks / occupiedCount : 0f;
        
        diagnostics.encoderTimeMs = System.currentTimeMillis() - startTime;
        
        return new EncodedState(tensor, null, null, null, config.stateEncodingSpec, diagnostics);
    }
    
    private boolean isNearStructure(int x, int y, int z, int w, int h, int f) {
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    int nx = x + dx;
                    int ny = y + dy;
                    int nz = z + dz;
                    if (nx >= 0 && nx < w && ny >= 0 && ny < h && nz >= 0 && nz < f) {
                        int idx = buffer.getIndex(nx, ny, nz);
                        if (idx != -1 && buffer.blockCounts[idx] > 0) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }
    
    private boolean isStructural(BuildingType type) {
        switch (type) {
            case FOUNDATION:
            case TRIANGLE_FOUNDATION:
            case WALL:
            case DOORWAY:
            case WINDOW_FRAME:
            case FLOOR:
            case TRIANGLE_FLOOR:
                return true;
            default:
                return false;
        }
    }
    
    private int mapTypeToChannel(BuildingType type) {
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
