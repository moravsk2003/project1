package com.rustbuilder.ai.rl.env.state;

import java.util.Arrays;

public class VoxelAggregationBuffer {
    public final int[] blockCounts;
    public final int[] typeMasks; // Stores bitmask of types present in this voxel
    public final float[] structuralStabilityMax;
    public final boolean[] hasTc;
    
    public final int width;
    public final int height;
    public final int floors;
    
    public VoxelAggregationBuffer(int width, int height, int floors) {
        this.width = width;
        this.height = height;
        this.floors = floors;
        
        int size = width * height * floors;
        this.blockCounts = new int[size];
        this.typeMasks = new int[size];
        this.structuralStabilityMax = new float[size];
        this.hasTc = new boolean[size];
    }
    
    public void reset() {
        Arrays.fill(blockCounts, 0);
        Arrays.fill(typeMasks, 0);
        Arrays.fill(structuralStabilityMax, 0f);
        Arrays.fill(hasTc, false);
    }
    
    public int getIndex(int x, int y, int z) {
        if (x < 0 || x >= width || y < 0 || y >= height || z < 0 || z >= floors) {
            return -1;
        }
        return z * (width * height) + y * width + x;
    }
}
