package com.rustbuilder.ai.rl;

import java.io.Serializable;

/**
 * Configuration for RL rewards and penalties.
 * Centralizes all reward shaping constants to avoid magic numbers.
 */
public class RLRewardConfig implements Serializable, Cloneable {
    private static final long serialVersionUID = 1L;

    // --- Step Rewards ---
    public double basePlacementReward = 0.08;
    public double socketConnectionReward = 0.09;
    public double socketConnectionMax = 0.99;
    public double disconnectedSegmentPenalty = -0.2;
    
    public double stabilityRewardMult = 0.04;
    public double floatingBlockPenalty = -0.35;
    
    public double tcPlacementBonus = 0.4;
    public double tcEnclosedBonus = 1.5;
    public double secondaryDeployableBonus = 0.15; // Workbench, LootRoom
    
    public double foundationCountBonus1 = 0.05;
    public double foundationCountBonus2 = 0.09;
    public double foundationCountBonus3 = 0.12;
    public double foundationCountBonus4 = 0.14;
    public double foundationCountBonus5 = 0.15;
    
    public double spatialCompactnessBonus = 0.07;
    public double spatialScatteredPenalty = -0.18;
    
    // --- Step Penalties (Invalid Actions) ---
    public double penaltyNoSupport = -0.35;
    public double penaltyBadSocket = -0.55;
    public double penaltyCollision = -0.25;
    public double penaltyFloorConstraint = -0.30;
    public double penaltyGenericInvalid = -0.20;

    // --- Growth Bonuses (Multi-Discrete Flow) ---
    public double blockGrowthReward = 0.15;
    public double growthStreakBonus = 0.03;
    public double noGrowthPenalty = -0.04;
    public double stopUnderbuildPenaltyLow = -0.5;
    public double stopUnderbuildPenaltyHigh = -1.0;
    public double stopRewardClampMin = -4.0;
    public double stopUnbuiltBlockPenalty = -0.01;

    // --- Final Evaluation Modifiers ---
    public double finalScoreMultiplier = 20.0;
    public double earlyStopPenaltyMult = -0.08;
    public double connectivityBonus = 0.1;
    public double fragmentBasePenalty = -0.2;
    public double fragmentDistPenaltyMult = -0.1;
    public double tcConnectivityPenalty = -0.01;
    public double tcDistancePenaltyMult = -0.01;
    public double logisticsBonus = 10.0;
    public double raidBonusMultiplier = 30.0;
    public double totalFailurePenalty = -5.0;

    public static RLRewardConfig createDefault() {
        return new RLRewardConfig();
    }

    @Override
    public RLRewardConfig clone() {
        try {
            return (RLRewardConfig) super.clone();
        } catch (CloneNotSupportedException e) {
            throw new AssertionError();
        }
    }
}
