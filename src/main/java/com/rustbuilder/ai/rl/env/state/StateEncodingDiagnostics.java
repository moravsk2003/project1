package com.rustbuilder.ai.rl.env.state;

public class StateEncodingDiagnostics {
    public long encoderTimeMs;
    
    public String encoderVersion;
    public int voxelChannels;
    
    public int encodedBlocksCount;
    public int skippedOutOfBoundsBlocks;
    public float outOfBoundsRatio;
    
    public int occupiedVoxelsCount;
    public int multiBlockVoxelCount;
    public int sameTypeMultiBlockVoxelCount;
    public int maxBlocksPerVoxel;
    public float avgBlocksPerOccupiedVoxel;
    
    public int occupancyBinaryNonzero;
    public int occupancyDensityNonzero;
    public int stabilityNonzero;
    public int supportBelowNonzero;
    public int nearStructureNonzero;
    public int tcProximityNonzero;
    
    public GlobalFeatureDiagnostics globalDiagnostics;
}
