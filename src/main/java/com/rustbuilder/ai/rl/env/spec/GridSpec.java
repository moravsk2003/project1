package com.rustbuilder.ai.rl.env.spec;

public class GridSpec {
    public final int width;
    public final int height;
    public final int floors;
    public final int maxBlocks;
    public final int tileCount;
    public final int voxelCount;

    public GridSpec(int width, int height, int floors, int maxBlocks) {
        this.width = width;
        this.height = height;
        this.floors = floors;
        this.maxBlocks = maxBlocks;
        this.tileCount = width * height;
        this.voxelCount = width * height * floors;
    }
}
