package com.rustbuilder.ai.rl.env.spec;

public class ActionSpaceSpec {
    public final int typeCount;
    public final int floorCount;
    public final int tileCount;
    public final int rotationCount;
    public final int aimCount;
    public final TileIndexingMode tileIndexingMode;

    public ActionSpaceSpec(int typeCount, int floorCount, int tileCount, int rotationCount, int aimCount, TileIndexingMode tileIndexingMode) {
        this.typeCount = typeCount;
        this.floorCount = floorCount;
        this.tileCount = tileCount;
        this.rotationCount = rotationCount;
        this.aimCount = aimCount;
        this.tileIndexingMode = tileIndexingMode;
    }
}
