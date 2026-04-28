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
    public int invalidActions = 0;
    public int totalActions = 0;
    public int blocksPlaced = 0;
    public boolean hasTC = false;
    public boolean hasLootRoom = false;
    public EvaluationResult evaluationResult = null;
    
    public List<MultiDiscreteExperienceReplay.Transition> episodeTransitions = new ArrayList<>();
    
    public Map<PlacementError, Integer> errorStats = new EnumMap<>(PlacementError.class);
    public Map<ActionType, Integer> typeStats = new EnumMap<>(ActionType.class);
}
