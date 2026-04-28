package com.rustbuilder.ai.rl.multidiscrete;

import com.rustbuilder.model.GridModel;
import java.util.List;
import java.util.Random;

/**
 * [RL REDESIGN]
 * Updated Heuristic policy for 5-phase multi-discrete system.
 * Samples actions using deep conditional masks to ensure physical validity.
 */
public class HeuristicMultiDiscretePhasePolicy implements MultiDiscretePhasePolicy {

    private final Random random;

    public HeuristicMultiDiscretePhasePolicy() {
        this.random = new Random();
    }

    public HeuristicMultiDiscretePhasePolicy(Random random) {
        this.random = random;
    }

    @Override
    public MultiDiscreteAction chooseAction(MultiDiscretePhaseContext context, MultiDiscreteStateObserver observer) {
        GridModel grid = context.getGrid();
        int step = context.getStep();

        // 1. PHASE: TYPE
        List<Integer> validTypes = HeuristicMaskingUtils.getFeasibleTypes(grid, context.isHasTC(), context.isHasLootRoom(), step);
        if (step > 0 && !validTypes.contains(com.rustbuilder.ai.rl.legacy.ActionSpace.STOP_TYPE_INDEX)) {
            validTypes.add(com.rustbuilder.ai.rl.legacy.ActionSpace.STOP_TYPE_INDEX);
        } else if (validTypes.isEmpty()) {
            validTypes.add(com.rustbuilder.ai.rl.legacy.ActionSpace.STOP_TYPE_INDEX);
        }
        int type = validTypes.get(random.nextInt(validTypes.size()));
        if (observer != null) observer.observePhase(context, "TYPE", 0, type);

        if (type == com.rustbuilder.ai.rl.legacy.ActionSpace.STOP_TYPE_INDEX) {
            return new MultiDiscreteAction(type, 0, 0, 0, 12);
        }

        // 2. PHASE: FLOOR
        List<Integer> validFloors = HeuristicMaskingUtils.getValidFloors(grid, type, step);
        if (validFloors.isEmpty()) return new MultiDiscreteAction(com.rustbuilder.ai.rl.legacy.ActionSpace.STOP_TYPE_INDEX, 0, 0, 0, 12);
        int floor = validFloors.get(random.nextInt(validFloors.size()));
        if (observer != null) observer.observePhase(context, "FLOOR", 1, floor);

        // 3. PHASE: TILE
        List<Integer> validTiles = HeuristicMaskingUtils.getValidTiles(grid, type, floor, step);
        if (validTiles.isEmpty()) return new MultiDiscreteAction(com.rustbuilder.ai.rl.legacy.ActionSpace.STOP_TYPE_INDEX, 0, 0, 0, 12);
        int tileIndex = validTiles.get(random.nextInt(validTiles.size()));
        if (observer != null) observer.observePhase(context, "TILE", 2, tileIndex);

        // 4. PHASE: ROTATION
        List<Integer> validRotations = HeuristicMaskingUtils.getValidRotations(grid, type, floor, tileIndex);
        if (validRotations.isEmpty()) return new MultiDiscreteAction(com.rustbuilder.ai.rl.legacy.ActionSpace.STOP_TYPE_INDEX, 0, 0, 0, 12);
        int rotation = validRotations.get(random.nextInt(validRotations.size()));
        if (observer != null) observer.observePhase(context, "ROTATION", 3, rotation);

        // 5. PHASE: AIM
        List<Integer> validAimSectors = HeuristicMaskingUtils.getValidAimSectors(grid, type, floor, tileIndex, rotation);
        if (validAimSectors.isEmpty()) return new MultiDiscreteAction(com.rustbuilder.ai.rl.legacy.ActionSpace.STOP_TYPE_INDEX, 0, 0, 0, 12);
        int aimSector = validAimSectors.get(random.nextInt(validAimSectors.size()));
        if (observer != null) observer.observePhase(context, "AIM", 4, aimSector);

        MultiDiscreteAction action = new MultiDiscreteAction(type, floor, tileIndex, rotation, aimSector);
        if (observer != null) observer.onActionAssembled(context, action);

        return action;
    }
}
