package com.rustbuilder.ai.rl.domain;

import com.rustbuilder.ai.rl.domain.reward.RewardFormulaSet;
import java.io.Serializable;

/**
 * Configuration for RL rewards and penalties.
 * Centralizes all reward shaping constants to avoid magic numbers.
 */
public class RLRewardConfig implements Serializable, Cloneable {
    private static final long serialVersionUID = 1L;

    // --- Step Rewards ---
    public double basePlacementReward = 0.08;
    public double socketConnectionReward = 0.07;
    public double socketConnectionMax = 0.9;
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
    public double blockGrowthReward = 0.05;
    public double growthStreakBonus = 0.01;
    public double mainComponentGrowthReward = 0.02;
    public double componentCountIncreasePenalty = 0.025;
    public double tcProtectionDeltaReward = 0.2;
    public double tcProtectionEpisodeRewardCap = 1.0;
    public double noGrowthPenalty = -0.04;
    public double stepEvalDeltaMultiplier = 10.0;
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
    public RewardFormulaSet rewardFormulaSet = new RewardFormulaSet();

    public static RLRewardConfig createDefault() {
        return new RLRewardConfig();
    }

    public void copyFrom(RLRewardConfig other) {
        if (other == null) return;

        this.basePlacementReward = other.basePlacementReward;
        this.socketConnectionReward = other.socketConnectionReward;
        this.socketConnectionMax = other.socketConnectionMax;
        this.disconnectedSegmentPenalty = other.disconnectedSegmentPenalty;
        this.stabilityRewardMult = other.stabilityRewardMult;
        this.floatingBlockPenalty = other.floatingBlockPenalty;
        this.tcPlacementBonus = other.tcPlacementBonus;
        this.tcEnclosedBonus = other.tcEnclosedBonus;
        this.secondaryDeployableBonus = other.secondaryDeployableBonus;
        this.foundationCountBonus1 = other.foundationCountBonus1;
        this.foundationCountBonus2 = other.foundationCountBonus2;
        this.foundationCountBonus3 = other.foundationCountBonus3;
        this.foundationCountBonus4 = other.foundationCountBonus4;
        this.foundationCountBonus5 = other.foundationCountBonus5;
        this.spatialCompactnessBonus = other.spatialCompactnessBonus;
        this.spatialScatteredPenalty = other.spatialScatteredPenalty;
        this.penaltyNoSupport = other.penaltyNoSupport;
        this.penaltyBadSocket = other.penaltyBadSocket;
        this.penaltyCollision = other.penaltyCollision;
        this.penaltyFloorConstraint = other.penaltyFloorConstraint;
        this.penaltyGenericInvalid = other.penaltyGenericInvalid;
        this.blockGrowthReward = other.blockGrowthReward;
        this.growthStreakBonus = other.growthStreakBonus;
        this.mainComponentGrowthReward = other.mainComponentGrowthReward;
        this.componentCountIncreasePenalty = other.componentCountIncreasePenalty;
        this.tcProtectionDeltaReward = other.tcProtectionDeltaReward;
        this.tcProtectionEpisodeRewardCap = other.tcProtectionEpisodeRewardCap;
        this.noGrowthPenalty = other.noGrowthPenalty;
        this.stepEvalDeltaMultiplier = other.stepEvalDeltaMultiplier;
        this.stopUnderbuildPenaltyLow = other.stopUnderbuildPenaltyLow;
        this.stopUnderbuildPenaltyHigh = other.stopUnderbuildPenaltyHigh;
        this.stopRewardClampMin = other.stopRewardClampMin;
        this.stopUnbuiltBlockPenalty = other.stopUnbuiltBlockPenalty;
        this.finalScoreMultiplier = other.finalScoreMultiplier;
        this.earlyStopPenaltyMult = other.earlyStopPenaltyMult;
        this.connectivityBonus = other.connectivityBonus;
        this.fragmentBasePenalty = other.fragmentBasePenalty;
        this.fragmentDistPenaltyMult = other.fragmentDistPenaltyMult;
        this.tcConnectivityPenalty = other.tcConnectivityPenalty;
        this.tcDistancePenaltyMult = other.tcDistancePenaltyMult;
        this.logisticsBonus = other.logisticsBonus;
        this.raidBonusMultiplier = other.raidBonusMultiplier;
        this.totalFailurePenalty = other.totalFailurePenalty;
        this.rewardFormulaSet = other.rewardFormulaSet != null
            ? other.rewardFormulaSet.clone()
            : new RewardFormulaSet();
    }

    public java.util.Map<String, Double> toMap() {
        java.util.Map<String, Double> map = new java.util.LinkedHashMap<>();
        for (java.lang.reflect.Field field : this.getClass().getFields()) {
            if (field.getType() == double.class) {
                try {
                    map.put(field.getName(), field.getDouble(this));
                } catch (IllegalAccessException ignored) {
                }
            }
        }
        return map;
    }

    @Override
    public RLRewardConfig clone() {
        try {
            RLRewardConfig clone = (RLRewardConfig) super.clone();
            clone.rewardFormulaSet = rewardFormulaSet != null
                ? rewardFormulaSet.clone()
                : new RewardFormulaSet();
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new AssertionError();
        }
    }
}
