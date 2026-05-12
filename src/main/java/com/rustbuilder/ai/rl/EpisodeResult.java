package com.rustbuilder.ai.rl;

import com.rustbuilder.model.GridModel;
import com.rustbuilder.service.evaluator.HouseEvaluator.EvaluationResult;
import com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteExperienceReplay;
import com.rustbuilder.core.action.BuildAction.ActionType;
import com.rustbuilder.core.placement.PlacementError;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Helper to store episode outcome and statistics.
 */
public class EpisodeResult {
    public GridModel grid;
    public double accStepReward = 0;     // Sum of individual step rewards
    public double finalEvalReward = 0;    // Final score from HouseEvaluator at episode end
    public double earlyStopPenalty = 0;   // Final-score penalty applied only for explicit early STOP
    public double finalRewardRawScore = 0;
    public double finalRewardLogisticsBonus = 0;
    public double finalRewardRaidBonus = 0;
    public double finalRewardConnectivityBonus = 0;
    public double finalRewardTcEnclosedBonus = 0;
    public double finalRewardFragmentPenalty = 0;
    public double finalRewardTcPenalty = 0;
    public double finalRewardFailurePenalty = 0;
    public double finalRewardFormulaBonus = 0;
    public double evalLogisticsScore = 0;
    public double evalCostScore = 0;
    public double evalRaidScore = 0;
    public double evalWorkingAreaScore = 0;
    public double evalSafeZoneScore = 0;
    public int raidSulfurToTC = -1;
    public int componentCount = 0;
    public int mainComponentBlocks = 0;
    public boolean finalRewardHasTC = false;
    public boolean finalRewardTcEnclosed = false;
    public double stepRewardInvalidPenalty = 0;
    public double stepRewardBasePlacement = 0;
    public double stepRewardSocketConnection = 0;
    public double stepRewardDisconnectedPenalty = 0;
    public double stepRewardStability = 0;
    public double stepRewardFloatingPenalty = 0;
    public double stepRewardTypeBonus = 0;
    public double stepRewardFoundationBonus = 0;
    public double stepRewardSpatialCompactness = 0;
    public double stepRewardSpatialScatteredPenalty = 0;
    public double stepRewardFormulaBonus = 0;
    public double stepRewardGrowth = 0;
    public double stepRewardGrowthStreak = 0;
    public double stepRewardMainComponentDelta = 0;
    public double stepRewardFragmentationDelta = 0;
    public double stepRewardTcProtectionDelta = 0;
    public double stepRewardEvalDelta = 0;
    public double stepRewardNoGrowthPenalty = 0;
    public double stepRewardInvalidStreakPenalty = 0;
    public double stopTransitionReward = 0;
    public double stopTransitionEarlyPenalty = 0;
    public double stopTransitionUnbuiltPenalty = 0;
    public double stopTransitionUnderbuildPenalty = 0;
    public double stopTransitionClampAdjustment = 0;
    public int invalidActions = 0;
    public int totalActions = 0;
    public int blocksPlaced = 0;
    public boolean hasTC = false;
    public boolean hasLootRoom = false;
    public EvaluationResult evaluationResult = null;

    public com.rustbuilder.ai.rl.log.StopReason stopReason = com.rustbuilder.ai.rl.log.StopReason.UNKNOWN;

    // Performance and diagnostics
    public long totalEncoderTimeMs = 0;
    public long totalGlobalEncoderTimeMs = 0;
    public long perfEpisodeNs = 0;
    public long perfContextNs = 0;
    public long perfStateEncodeNs = 0;
    public long perfNextStateEncodeNs = 0;
    public long perfActionSelectNs = 0;
    public long perfGridCloneNs = 0;
    public long perfPlacementNs = 0;
    public long perfFinalizeNs = 0;
    public long perfRewardNs = 0;
    public long perfReplayNs = 0;
    public long perfTrainNs = 0;
    public long perfInvalidLogNs = 0;
    public long perfFinalEvalNs = 0;
    public long perfEpisodeLogNs = 0;
    public long perfUpdateBestNs = 0;
    public com.rustbuilder.ai.rl.env.state.GlobalFeatureDiagnostics globalDiag = null;

    public List<MultiDiscreteExperienceReplay.Transition> episodeTransitions = new ArrayList<>();

    public Map<PlacementError, Integer> errorStats = new EnumMap<>(PlacementError.class);
    public Map<ActionType, Integer> typeStats = new EnumMap<>(ActionType.class);

    public void addStepRewardBreakdown(StepRewardFunction.Breakdown breakdown) {
        if (breakdown == null) return;
        stepRewardInvalidPenalty += breakdown.invalidPenalty;
        stepRewardBasePlacement += breakdown.basePlacement;
        stepRewardSocketConnection += breakdown.socketConnection;
        stepRewardDisconnectedPenalty += breakdown.disconnectedPenalty;
        stepRewardStability += breakdown.stabilityReward;
        stepRewardFloatingPenalty += breakdown.floatingPenalty;
        stepRewardTypeBonus += breakdown.typeBonus;
        stepRewardFoundationBonus += breakdown.foundationBonus;
        stepRewardSpatialCompactness += breakdown.spatialCompactness;
        stepRewardSpatialScatteredPenalty += breakdown.spatialScatteredPenalty;
        stepRewardFormulaBonus += breakdown.formulaReward;
    }

    public double stepRewardBreakdownTotal() {
        return stepRewardInvalidPenalty
            + stepRewardBasePlacement
            + stepRewardSocketConnection
            + stepRewardDisconnectedPenalty
            + stepRewardStability
            + stepRewardFloatingPenalty
            + stepRewardTypeBonus
            + stepRewardFoundationBonus
            + stepRewardSpatialCompactness
            + stepRewardSpatialScatteredPenalty
            + stepRewardFormulaBonus
            + stepRewardGrowth
            + stepRewardGrowthStreak
            + stepRewardMainComponentDelta
            + stepRewardFragmentationDelta
            + stepRewardTcProtectionDelta
            + stepRewardEvalDelta
            + stepRewardNoGrowthPenalty
            + stepRewardInvalidStreakPenalty;
    }

    public String stepRewardBreakdownSummary() {
        return String.format(Locale.US,
            "place %.2f, socket %.2f, stab %.2f, type %.2f, found %.2f, spatial %.2f, formula %.2f, growth %.2f, streak %.2f, mainComp %.2f, fragDelta %.2f, tcProtect %.2f, evalDelta %.2f, invalid %.2f, noGrowth %.2f, stopTrans %.2f",
            stepRewardBasePlacement,
            stepRewardSocketConnection,
            stepRewardStability + stepRewardFloatingPenalty,
            stepRewardTypeBonus,
            stepRewardFoundationBonus,
            stepRewardSpatialCompactness + stepRewardSpatialScatteredPenalty + stepRewardDisconnectedPenalty,
            stepRewardFormulaBonus,
            stepRewardGrowth,
            stepRewardGrowthStreak,
            stepRewardMainComponentDelta,
            stepRewardFragmentationDelta,
            stepRewardTcProtectionDelta,
            stepRewardEvalDelta,
            stepRewardInvalidPenalty + stepRewardInvalidStreakPenalty,
            stepRewardNoGrowthPenalty,
            stopTransitionReward);
    }

    public void resetFinalRewardBreakdown() {
        finalEvalReward = 0;
        earlyStopPenalty = 0;
        finalRewardRawScore = 0;
        finalRewardLogisticsBonus = 0;
        finalRewardRaidBonus = 0;
        finalRewardConnectivityBonus = 0;
        finalRewardTcEnclosedBonus = 0;
        finalRewardFragmentPenalty = 0;
        finalRewardTcPenalty = 0;
        finalRewardFailurePenalty = 0;
        finalRewardFormulaBonus = 0;
        evalLogisticsScore = 0;
        evalCostScore = 0;
        evalRaidScore = 0;
        evalWorkingAreaScore = 0;
        evalSafeZoneScore = 0;
        raidSulfurToTC = -1;
        componentCount = 0;
        mainComponentBlocks = 0;
        finalRewardHasTC = false;
        finalRewardTcEnclosed = false;
    }

    public String finalRewardBreakdownSummary() {
        if (finalRewardFailurePenalty != 0.0) {
            return String.format(Locale.US,
                "failure %.2f + early %.2f",
                finalRewardFailurePenalty, earlyStopPenalty);
        }

        return String.format(Locale.US,
            "raw %.2f, log %.2f, raid %.2f, conn %.2f, tcBox %.2f, formula %.2f, early %.2f, frag %.2f, tc %.2f",
            finalRewardRawScore,
            finalRewardLogisticsBonus,
            finalRewardRaidBonus,
            finalRewardConnectivityBonus,
            finalRewardTcEnclosedBonus,
            finalRewardFormulaBonus,
            earlyStopPenalty,
            finalRewardFragmentPenalty,
            finalRewardTcPenalty);
    }
}
