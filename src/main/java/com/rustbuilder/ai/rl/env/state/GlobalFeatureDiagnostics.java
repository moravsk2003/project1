package com.rustbuilder.ai.rl.env.state;

public class GlobalFeatureDiagnostics {
    public int globalFeatureCount;
    public int globalNonzeroCount;
    public float globalMin;
    public float globalMax;
    public float globalMean;
    
    public float blockCountNorm;
    public float occupiedVoxelCountNorm;
    public float multiBlockVoxelRatio;
    public float tcPresent;
    public float maxFloorNorm;
    public float avgStructuralStability;
    public float outOfBoundsRatio;
    
    public boolean hasNaN = false;
    public boolean hasInfinity = false;
}
