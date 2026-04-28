package com.rustbuilder.service.evaluator;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.ResourceType;

/**
 * Evaluates total resource cost of all placed blocks.
 * Score is normalized: lower cost = better score.
 */
public class ResourceCostEvaluator {

    // Reference cost for normalization (typical small base)
    // Increased to 35000 so the AI isn't overly punished for placing walls
    private static final double BASE_COST = 35000.0;

    public static class CostResult {
        public final Map<ResourceType, Integer> totalCost;
        public final double score; // 0.0 - 1.0, higher = cheaper

        public CostResult(Map<ResourceType, Integer> totalCost, double score) {
            this.totalCost = totalCost;
            this.score = score;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder("Resources: ");
            totalCost.forEach((k, v) -> sb.append(k).append("=").append(v).append(" "));
            sb.append("| Score=").append(String.format("%.2f", score));
            return sb.toString();
        }
    }

    /**
     * Evaluates total resource cost and scales it by base quality.
     * costScore = costBase * quality.
     * This ensures that cost is mainly important for high-quality bases, 
     * and doesn't prevent building larger quality bases in the early stages.
     */
    public CostResult evaluate(List<BuildingBlock> blocks, double logisticsScore, double raidScore, double workingAreaScore) {
        Map<ResourceType, Integer> totalCost = new HashMap<>();

        for (BuildingBlock block : blocks) {
            Map<ResourceType, Integer> blockCost = block.getBuildCost();
            blockCost.forEach((res, amount) -> totalCost.merge(res, amount, Integer::sum));
        }

        // Convert to a single comparable number (weighted sum)
        double weightedTotal = 0;
        for (Map.Entry<ResourceType, Integer> entry : totalCost.entrySet()) {
            double weight = getResourceWeight(entry.getKey());
            weightedTotal += entry.getValue() * weight;
        }

        // costBase: 1 / (1 + total/BASE_COST). Higher score = cheaper.
        double costBase = 1.0 / (1.0 + weightedTotal / BASE_COST);

        // Quality factor Q based on performance criteria
        double q = (logisticsScore + raidScore + workingAreaScore) / 3.0;
        q = Math.max(0.0, Math.min(1.0, q));

        // Final score: if base is bad, cost is ignored (low Q). 
        // If base is good, cost becomes a major factor.
        double score = costBase * q;

        return new CostResult(totalCost, score);
    }

    /**
     * Relative "value" weight for each resource type.
     * Higher tier resources are more expensive to gather.
     */
    private double getResourceWeight(ResourceType type) {
        switch (type) {
            case WOOD:  return 1.0;
            case STONE: return 1.0;
            case METAL: return 2.5; 
            case HQM:   return 10.0;
            default:    return 1.0;
        }
    }
}
