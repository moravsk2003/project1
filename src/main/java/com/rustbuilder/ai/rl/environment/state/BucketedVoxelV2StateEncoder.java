package com.rustbuilder.ai.rl.environment.state;


import com.rustbuilder.ai.rl.environment.spec.GridSpec;
import com.rustbuilder.ai.rl.environment.spec.StateEncodingSpec;
import com.rustbuilder.core.action.BuildAction.ActionType;
import com.rustbuilder.ai.rl.environment.spec.CoordinateEncodingMode;
import com.rustbuilder.ai.rl.environment.spec.EncodingRuntimeConfig;
import com.rustbuilder.config.GameConstants;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.List;

public class BucketedVoxelV2StateEncoder implements StateRepresentationEncoder {

    // Voxel tensor channel indices (channels 0-9 are type masks, controlled via loop variable)
    private static final int CHANNEL_OCCUPANCY_BINARY = 10;
    private static final int CHANNEL_DENSITY = 11;
    private static final int CHANNEL_STABILITY = 12;
    private static final int CHANNEL_SUPPORT_BELOW = 13;
    private static final int CHANNEL_NEAR_STRUCTURE = 14;
    private static final int CHANNEL_TC_PROXIMITY = 15;

    private final EncodingRuntimeConfig config;
    private final VoxelAggregationBuffer buffer;
    
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
            int gx = (int) Math.round((b.getX() + GameConstants.HALF_TILE - GameConstants.GRID_ORIGIN_X) / GameConstants.TILE_SIZE);
            int gy = (int) Math.round((b.getY() + GameConstants.HALF_TILE - GameConstants.GRID_ORIGIN_Y) / GameConstants.TILE_SIZE);
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
            
        int channels = 16;
        float[] flatTensor = new float[channels * gridF * gridH * gridW];
        
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
                                flatTensor[tensorIndex(c, z, y, x, gridF, gridH, gridW)] = 1.0f;
                            }
                        }
                        
                        flatTensor[tensorIndex(CHANNEL_OCCUPANCY_BINARY, z, y, x, gridF, gridH, gridW)] = 1.0f;
                        diagnostics.occupancyBinaryNonzero++;
                        
                        double density = Math.min(count / 4.0, 1.0);
                        flatTensor[tensorIndex(CHANNEL_DENSITY, z, y, x, gridF, gridH, gridW)] = (float) density;
                        diagnostics.occupancyDensityNonzero++;
                        
                        if (buffer.structuralStabilityMax[idx] > 0) {
                            flatTensor[tensorIndex(CHANNEL_STABILITY, z, y, x, gridF, gridH, gridW)] = buffer.structuralStabilityMax[idx];
                            diagnostics.stabilityNonzero++;
                        }
                    }
                    
                    if (z == 0 || (idx >= gridW * gridH && buffer.blockCounts[idx - gridW * gridH] > 0)) {
                        flatTensor[tensorIndex(CHANNEL_SUPPORT_BELOW, z, y, x, gridF, gridH, gridW)] = 1.0f;
                        diagnostics.supportBelowNonzero++;
                    }
                    
                    if (isNearStructure(x, y, z, gridW, gridH, gridF)) {
                        flatTensor[tensorIndex(CHANNEL_NEAR_STRUCTURE, z, y, x, gridF, gridH, gridW)] = 1.0f;
                        diagnostics.nearStructureNonzero++;
                    }
                    
                    if (hasAnyTc) {
                        double dx = x - tcX, dy = y - tcY, dz = z - tcZ;
                        double dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
                        double prox = 1.0 - Math.min(dist / maxDist, 1.0);
                        if (prox > 0) {
                            flatTensor[tensorIndex(CHANNEL_TC_PROXIMITY, z, y, x, gridF, gridH, gridW)] = (float) prox;
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
        
        INDArray tensor = Nd4j.create(flatTensor, new int[]{1, channels, gridF, gridH, gridW}, 'c');
        return new EncodedState(tensor, null, null, null, config.stateEncodingSpec, diagnostics);
    }

    private static int tensorIndex(int c, int z, int y, int x, int floors, int height, int width) {
        return (((c * floors) + z) * height + y) * width + x;
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
