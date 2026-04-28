package com.rustbuilder.ai.rl.multidiscrete;

import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import com.rustbuilder.model.GridModel;

/**
 * Enhanced experience replay with balanced sampling.
 * Maintains separate pools for successful (valid) and unsuccessful (invalid) transitions
 * to ensure the agent doesn't over-learn from early randomized failure noise.
 */
public class MultiDiscreteExperienceReplay {

    public static class Transition {
        public INDArray state;
        public MultiDiscreteAction action;
        public double reward;
        public INDArray nextState;
        public boolean isDone;
        public int step; 
        
        // Snapshots of the grid at current and next state for real-time masking during trainBatch
        public GridModel grid;
        public GridModel nextGrid;
        public boolean isSuccess;

        public Transition(INDArray state, MultiDiscreteAction action, double reward, 
                          INDArray nextState, boolean isDone, int step,
                          GridModel grid, GridModel nextGrid, boolean isSuccess) {
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.nextState = nextState;
            this.isDone = isDone;
            this.step = step;
            this.grid = grid;
            this.nextGrid = nextGrid;
            this.isSuccess = isSuccess;
        }

        public void close() {
            if (state != null && !state.wasClosed()) state.close();
            if (nextState != null && !nextState.wasClosed()) nextState.close();
        }
    }

    private final int capacity;
    private final LinkedList<Transition> validBuffer;
    private final LinkedList<Transition> invalidBuffer;
    private final Random random;

    public MultiDiscreteExperienceReplay(int capacity) {
        this.capacity = capacity;
        this.validBuffer = new LinkedList<>();
        this.invalidBuffer = new LinkedList<>();
        this.random = new Random();
    }

    public void add(Transition transition) {
        boolean isValid = transition.isSuccess;
        LinkedList<Transition> targetBuffer = isValid ? validBuffer : invalidBuffer;

        if (targetBuffer.size() >= capacity / 2) {
            Transition old = targetBuffer.removeFirst();
            if (old != null) old.close();
        }
        targetBuffer.addLast(transition);
    }

    /**
     * Samples a balanced batch (approx. 50% valid, 50% invalid if possible).
     */
    public List<Transition> sample(int batchSize) {
        List<Transition> sample = new ArrayList<>(batchSize);
        if (size() == 0) return sample;

        int validCount = Math.min(batchSize / 2, validBuffer.size());
        int invalidCount = batchSize - validCount;

        // If one buffer is too small, fill from the other
        if (invalidBuffer.size() < invalidCount) {
            invalidCount = invalidBuffer.size();
            validCount = Math.min(batchSize - invalidCount, validBuffer.size());
        }

        for (int i = 0; i < validCount; i++) {
            sample.add(validBuffer.get(random.nextInt(validBuffer.size())));
        }
        for (int i = 0; i < invalidCount; i++) {
            sample.add(invalidBuffer.get(random.nextInt(invalidBuffer.size())));
        }

        return sample;
    }

    public int size() {
        return validBuffer.size() + invalidBuffer.size();
    }

    public void clear() {
        for (Transition t : validBuffer) if (t != null) t.close();
        for (Transition t : invalidBuffer) if (t != null) t.close();
        validBuffer.clear();
        invalidBuffer.clear();
    }
}
