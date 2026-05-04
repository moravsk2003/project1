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

        // 1. PHASE: TYPE
        List<Integer> validTypes = HeuristicMaskingUtils.getFeasibleTypes(context);
        int blockCount = context.getBlockCount();
        if (blockCount >= MultiDiscreteActionSpace.MIN_BLOCKS_BEFORE_STOP
                && !validTypes.contains(MultiDiscreteActionSpace.STOP_TYPE_INDEX)) {
            validTypes.add(MultiDiscreteActionSpace.STOP_TYPE_INDEX);
        } else if (validTypes.isEmpty()) {
            return null;
        }
        int type = validTypes.get(random.nextInt(validTypes.size()));
        if (observer != null) observer.observePhase(context, "TYPE", 0, type);

        if (type == MultiDiscreteActionSpace.STOP_TYPE_INDEX) {
            return new MultiDiscreteAction(type, 0, 0, 0, MultiDiscreteActionSpace.DEFAULT_AIM_SECTOR);
        }

        // 2. PHASE: FLOOR
        List<Integer> validFloors = HeuristicMaskingUtils.getValidFloors(context, type);
        if (validFloors.isEmpty()) return null;
        int floor = validFloors.get(random.nextInt(validFloors.size()));
        if (observer != null) observer.observePhase(context, "FLOOR", 1, floor);

        // 3. PHASE: TILE
        List<Integer> validTiles = HeuristicMaskingUtils.getValidTiles(context, type, floor);
        if (validTiles.isEmpty()) return null;
        int tileIndex = validTiles.get(random.nextInt(validTiles.size()));
        if (observer != null) observer.observePhase(context, "TILE", 2, tileIndex);

        // 4. PHASE: ROTATION
        List<Integer> validRotations = HeuristicMaskingUtils.getValidRotations(grid, type, floor, tileIndex);
        if (validRotations.isEmpty()) return null;
        int rotation = validRotations.get(random.nextInt(validRotations.size()));
        if (observer != null) observer.observePhase(context, "ROTATION", 3, rotation);

        // 5. PHASE: AIM
        int aimSector = HeuristicMaskingUtils.getFirstValidAimSector(grid, type, floor, tileIndex, rotation);
        if (aimSector < 0) {
            MultiDiscreteAction fallback = HeuristicMaskingUtils.findFeasibleBuildAction(context, random);
            if (fallback != null) {
                if (observer != null) observer.onActionAssembled(context, fallback);
                return fallback;
            }
            if (blockCount >= MultiDiscreteActionSpace.MIN_BLOCKS_BEFORE_STOP) {
                return new MultiDiscreteAction(
                    MultiDiscreteActionSpace.STOP_TYPE_INDEX,
                    0,
                    0,
                    0,
                    MultiDiscreteActionSpace.DEFAULT_AIM_SECTOR
                );
            }
            return null;
        }
        if (observer != null) observer.observePhase(context, "AIM", 4, aimSector);

        MultiDiscreteAction action = new MultiDiscreteAction(type, floor, tileIndex, rotation, aimSector);
        if (observer != null) observer.onActionAssembled(context, action);

        return action;
    }

    private MultiDiscreteAction fallbackAction(MultiDiscretePhaseContext context, MultiDiscreteStateObserver observer) {
        return null;
    }
}
