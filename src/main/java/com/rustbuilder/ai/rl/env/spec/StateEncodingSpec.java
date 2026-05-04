package com.rustbuilder.ai.rl.env.spec;

public class StateEncodingSpec {
    public final String encoderName;
    public final String encoderVersion;
    public final int voxelChannels;
    public final int[] gridShape;
    public final boolean hasGlobalVector;
    public final int globalFeatureCount;
    public final boolean hasObjectTable;
    public final boolean hasGraphState;

    public StateEncodingSpec(String encoderName, String encoderVersion, int voxelChannels, int[] gridShape, boolean hasGlobalVector, int globalFeatureCount, boolean hasObjectTable, boolean hasGraphState) {
        this.encoderName = encoderName;
        this.encoderVersion = encoderVersion;
        this.voxelChannels = voxelChannels;
        this.gridShape = gridShape;
        this.hasGlobalVector = hasGlobalVector;
        this.globalFeatureCount = globalFeatureCount;
        this.hasObjectTable = hasObjectTable;
        this.hasGraphState = hasGraphState;
    }
}
