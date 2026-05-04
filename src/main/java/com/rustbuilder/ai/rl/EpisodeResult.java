package com.rustbuilder.ai.rl;

import com.rustbuilder.model.GridModel;
import com.rustbuilder.service.evaluator.HouseEvaluator.EvaluationResult;
import com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteExperienceReplay;
import com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * Helper to store episode outcome and statistics.
 */
public class EpisodeResult {
    public GridModel grid;
    public double accStepReward = 0;     // Sum of individual step rewards
    public double finalEvalReward = 0;    // Final score from HouseEvaluator at episode end
    public double earlyStopPenalty = 0;   // Final-score penalty applied only for explicit early STOP
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
}
