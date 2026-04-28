package com.rustbuilder.ai.rl.multidiscrete;

import com.rustbuilder.model.GridModel;

/**
 * [RL REDESIGN]
 * Immutable context containing all data required for a multi-discrete phase decision.
 */
public class MultiDiscretePhaseContext {
    private final GridModel grid;
    private final boolean hasTC;
    private final boolean hasLootRoom;
    private final int step;
    private final int maxSteps;

    public MultiDiscretePhaseContext(GridModel grid, boolean hasTC, boolean hasLootRoom, int step, int maxSteps) {
        this.grid = grid;
        this.hasTC = hasTC;
        this.hasLootRoom = hasLootRoom;
        this.step = step;
        this.maxSteps = maxSteps;
    }

    public GridModel getGrid() {
        return grid;
    }

    public boolean isHasTC() {
        return hasTC;
    }

    public boolean isHasLootRoom() {
        return hasLootRoom;
    }

    public int getStep() {
        return step;
    }

    public int getMaxSteps() {
        return maxSteps;
    }
}
