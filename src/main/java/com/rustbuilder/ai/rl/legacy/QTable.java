package com.rustbuilder.ai.rl.legacy;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Q-Table implementation for Q-Learning.
 * Maps state hash to arrays of expected Q-values per action.
 */
public class QTable implements Serializable {

    private static final long serialVersionUID = 1L;
    
    // Map of: State Hash -> Q-Value Array (size ACTION_COUNT)
    private final Map<Long, float[]> table = new HashMap<>();
    private final int actionCount;
    private final Random rng = new Random();

    public QTable() {
        this(ActionSpace.TOTAL_ACTIONS);
    }

    public QTable(int actionCount) {
        this.actionCount = actionCount;
    }
    
    // Action value index mapping (since STOP is -1, we store it at ACTION_COUNT)
    private int getArrayIndex(int actionIndex) {
        if (actionIndex == ActionSpace.STOP_TYPE_INDEX && actionCount == ActionSpace.TOTAL_ACTIONS) {
             return actionCount - 1; 
        }
        return actionIndex;
    }

    private float[] getOrCreateQValues(long state) {
        return table.computeIfAbsent(state, k -> new float[actionCount]);
    }

    /**
     * Gets the expected Q-value for a specific state-action pair.
     */
    public float getQValue(long state, int action) {
        float[] values = table.get(state);
        if (values == null) return 0.0f;
        return values[getArrayIndex(action)];
    }

    /**
     * Updates the Q-Value using the Bellman equation.
     */
    public void updateQValue(long state, int action, double reward, long nextState, double alpha, double gamma) {
        float[] currentQArray = getOrCreateQValues(state);
        int aIdx = getArrayIndex(action);
        
        float currentQ = currentQArray[aIdx];
        float maxNextQ = getMaxQValue(nextState); // greedy next action
        
        float newQ = (float) (currentQ + alpha * (reward + gamma * maxNextQ - currentQ));
        currentQArray[aIdx] = newQ;
    }

    /**
     * Gets highest Q-Value possible from the given state.
     */
    public float getMaxQValue(long state) {
        float[] values = table.get(state);
        if (values == null) return 0.0f;
        
        float max = -Float.MAX_VALUE;
        for (float v : values) {
            if (v > max) max = v;
        }
        // If uninitialized, everything is 0.
        return max == -Float.MAX_VALUE ? 0.0f : max;
    }

    /**
     * Epsilon-greedy action selection.
     */
    public int selectAction(long state, List<Integer> validActions, double epsilon) {
        if (validActions.isEmpty()) return ActionSpace.STOP_TYPE_INDEX;

        // Explore
        if (rng.nextDouble() < epsilon) {
            return validActions.get(rng.nextInt(validActions.size()));
        }

        // Exploit
        float[] qValues = table.get(state);
        if (qValues == null) {
            // Unseen state, pick randomly to break ties
            return validActions.get(rng.nextInt(validActions.size()));
        }

        int bestAction = validActions.get(0);
        float bestQ = -Float.MAX_VALUE;
        
        // Small tie breaker noise so it doesn't get stuck doing same 0.0 actions
        for (int a : validActions) {
            float q = qValues[getArrayIndex(a)];
            if (q > bestQ) {
                bestQ = q;
                bestAction = a;
            } else if (q == bestQ && rng.nextBoolean()) {
                bestAction = a; // break ties randomly
            }
        }
        return bestAction;
    }

    public int size() {
        return table.size();
    }
}
