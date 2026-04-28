package com.rustbuilder.ai.rl.nn;

import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ExperienceReplay {
    
    public static class Transition {
        public INDArray state;
        public int action;
        public double reward;
        public INDArray nextState;
        public boolean isDone;
        public List<Integer> validNextActions;

        public Transition(INDArray state, int action, double reward, INDArray nextState, boolean isDone, List<Integer> validNextActions) {
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.nextState = nextState;
            this.isDone = isDone;
            this.validNextActions = validNextActions;
        }

        public void close() {
            if (state != null && !state.wasClosed()) state.close();
            if (nextState != null && !nextState.wasClosed()) nextState.close();
        }
    }

    private final int capacity;
    private final List<Transition> memory;
    private final Random random;

    public ExperienceReplay(int capacity) {
        this.capacity = capacity;
        this.memory = new ArrayList<>(capacity);
        this.random = new Random();
    }

    public void add(Transition transition) {
        if (memory.size() >= capacity) {
            Transition old = memory.remove(0); // Pop oldest
            if (old != null) old.close();
        }
        memory.add(transition);
    }

    public List<Transition> sample(int batchSize) {
        List<Transition> batch = new ArrayList<>(batchSize);
        int memSize = memory.size();
        for (int i = 0; i < batchSize; i++) {
            batch.add(memory.get(random.nextInt(memSize)));
        }
        return batch;
    }

    public int size() {
        return memory.size();
    }
}
