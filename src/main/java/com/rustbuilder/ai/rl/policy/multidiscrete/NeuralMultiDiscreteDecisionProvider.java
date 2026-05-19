package com.rustbuilder.ai.rl.policy.multidiscrete;

import com.rustbuilder.ai.rl.environment.state.EncodedState;
import com.rustbuilder.ai.rl.environment.state.StateRepresentationEncoder;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.Function;

/**
 * Provides multi-discrete actions using a trained neural network with conditional head architecture.
 * Uses real-time conditional masking to ensure high-quality valid actions during both exploration and exploitation.
 */
public class NeuralMultiDiscreteDecisionProvider implements MultiDiscretePhaseDecisionProvider {

    private final MultiDiscreteDQNAgent agent;
    private final StateRepresentationEncoder stateEncoder;
    private final Random random;
    private double epsilon = 0.1;
    private boolean useAimSectorLearning = false;

    public NeuralMultiDiscreteDecisionProvider(MultiDiscreteDQNAgent agent, StateRepresentationEncoder stateEncoder) {
        this.agent = Objects.requireNonNull(agent, "Agent cannot be null");
        this.stateEncoder = stateEncoder;
        this.random = new Random();
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    public void setUseAimSectorLearning(boolean useAimSectorLearning) {
        this.useAimSectorLearning = useAimSectorLearning;
    }

    @Override
    public MultiDiscretePhaseDecision provideDecision(MultiDiscretePhaseContext context) {
        EncodedState state = null;
        List<INDArray> ownedArrays = new ArrayList<>();
        try {
            state = stateEncoder.encode(context.getGrid(), -1);
            int[] selected = new int[5];

            INDArray stateFeatures = agent.encodeStateFeatures(state.voxelTensor(), state.getGlobalVector());
            ownedArrays.add(stateFeatures);

            // Roll explore vs exploit once for the entire action
            boolean explore = random.nextDouble() < epsilon;

            // 1. PHASE: TYPE (Global step constraints)
            List<Integer> validTypes = HeuristicMaskingUtils.getFeasibleTypes(context);

            int blockCount = context.getBlockCount();
            if (blockCount >= MultiDiscreteActionSpace.MIN_BLOCKS_BEFORE_STOP
                    && !validTypes.contains(MultiDiscreteActionSpace.STOP_TYPE_INDEX)) {
                validTypes.add(MultiDiscreteActionSpace.STOP_TYPE_INDEX);
            }

            if (validTypes.isEmpty()) {
                return null;
            }

            if (explore) {
                selected[0] = validTypes.get(random.nextInt(validTypes.size()));
            } else {
                INDArray typeLogits = agent.predictType(stateFeatures);
                ownedArrays.add(typeLogits);
                selected[0] = maskedArgmax(typeLogits, 0, validTypes);
            }

            if (selected[0] == MultiDiscreteActionSpace.STOP_TYPE_INDEX) {
                return new MultiDiscretePhaseDecision(selected[0], 0, 0, 0, MultiDiscreteActionSpace.DEFAULT_AIM_SECTOR);
            }

            // 2. PHASE: FLOOR (Conditional on Type)
            List<Integer> validFloors = HeuristicMaskingUtils.getValidFloors(context, selected[0]);
            if (validFloors.isEmpty()) {
                return null;
            }

            if (explore) {
                selected[1] = validFloors.get(random.nextInt(validFloors.size()));
            } else {
                selected[1] = predictHeadAndTrack(
                    stateFeatures,
                    new int[]{selected[0]},
                    new int[]{MultiDiscreteActionSpace.TYPE_COUNT},
                    agent::predictFloor, validFloors, ownedArrays
                );
            }

            // 3. PHASE: TILE (Conditional on Type, Floor)
            List<Integer> validTiles = HeuristicMaskingUtils.getValidTiles(context, selected[0], selected[1]);
            if (validTiles.isEmpty()) {
                return null;
            }

            if (explore) {
                selected[2] = validTiles.get(random.nextInt(validTiles.size()));
            } else {
                selected[2] = predictHeadAndTrack(
                    stateFeatures,
                    new int[]{selected[0], selected[1]},
                    new int[]{MultiDiscreteActionSpace.TYPE_COUNT, MultiDiscreteActionSpace.FLOOR_COUNT},
                    agent::predictTile, validTiles, ownedArrays
                );
            }

            // 4. PHASE: ROTATION (Conditional on Type, Floor, Tile)
            List<Integer> validRotations = HeuristicMaskingUtils.getValidRotations(context.getGrid(), selected[0], selected[1], selected[2]);
            if (validRotations.isEmpty()) {
                return null;
            }

            if (explore) {
                selected[3] = validRotations.get(random.nextInt(validRotations.size()));
            } else {
                selected[3] = predictHeadAndTrack(
                    stateFeatures,
                    new int[]{selected[0], selected[1], selected[2]},
                    new int[]{MultiDiscreteActionSpace.TYPE_COUNT, MultiDiscreteActionSpace.FLOOR_COUNT, MultiDiscreteActionSpace.TILE_COUNT},
                    agent::predictRot, validRotations, ownedArrays
                );
            }

            // 5. PHASE: AIM (Conditional on Type, Floor, Tile, Rotation)
            if (!useAimSectorLearning) {
                int aim = HeuristicMaskingUtils.getFirstValidAimSector(context.getGrid(), selected[0], selected[1], selected[2], selected[3]);
                if (aim < 0) {
                    return fallbackDecision(context, blockCount);
                } else {
                    selected[4] = aim;
                }
            } else {
                List<Integer> validAimSectors = HeuristicMaskingUtils.getValidAimSectors(context.getGrid(), selected[0], selected[1], selected[2], selected[3]);
                if (validAimSectors.isEmpty()) {
                    return fallbackDecision(context, blockCount);
                } else if (explore) {
                    selected[4] = validAimSectors.get(random.nextInt(validAimSectors.size()));
                } else {
                    selected[4] = predictHeadAndTrack(
                        stateFeatures,
                        new int[]{selected[0], selected[1], selected[2], selected[3]},
                        new int[]{MultiDiscreteActionSpace.TYPE_COUNT, MultiDiscreteActionSpace.FLOOR_COUNT, MultiDiscreteActionSpace.TILE_COUNT, MultiDiscreteActionSpace.ROTATION_COUNT},
                        agent::predictAim, validAimSectors, ownedArrays
                    );
                }
            }

            return new MultiDiscretePhaseDecision(selected[0], selected[1], selected[2], selected[3], selected[4]);
        } finally {
            closeAll(ownedArrays);
            if (state != null && !state.wasClosed()) {
                state.close();
            }
        }
    }

    /**
     * Creates one-hot context vectors from previous action selections, concatenates them
     * with state features, runs the prediction head, and returns the masked argmax.
     * All created INDArrays are tracked in ownedArrays for proper off-heap cleanup.
     */
    private int predictHeadAndTrack(
            INDArray stateFeatures,
            int[] previousSelections,
            int[] contextSizes,
            Function<INDArray, INDArray> predictFunc,
            List<Integer> validOptions,
            List<INDArray> ownedArrays) {
        INDArray[] contexts = new INDArray[previousSelections.length];
        for (int i = 0; i < previousSelections.length; i++) {
            contexts[i] = ActionConditioningUtils.oneHot(previousSelections[i], contextSizes[i]);
            ownedArrays.add(contexts[i]);
        }
        INDArray input = ActionConditioningUtils.concat(stateFeatures, contexts);
        ownedArrays.add(input);
        INDArray logits = predictFunc.apply(input);
        ownedArrays.add(logits);
        return maskedArgmax(logits, 0, validOptions);
    }

    private MultiDiscretePhaseDecision fallbackDecision(MultiDiscretePhaseContext context, int blockCount) {
        MultiDiscreteAction fallback = HeuristicMaskingUtils.findFeasibleBuildAction(context, random);
        if (fallback != null) {
            return new MultiDiscretePhaseDecision(
                fallback.getTypeIndex(),
                fallback.getFloorIndex(),
                fallback.getTileIndex(),
                fallback.getRotationIndex(),
                fallback.getAimSector()
            );
        }
        if (blockCount >= MultiDiscreteActionSpace.MIN_BLOCKS_BEFORE_STOP) {
            return new MultiDiscretePhaseDecision(
                MultiDiscreteActionSpace.STOP_TYPE_INDEX,
                0,
                0,
                0,
                MultiDiscreteActionSpace.DEFAULT_AIM_SECTOR
            );
        }
        return null;
    }

    private void closeAll(List<INDArray> arrays) {
        for (int i = arrays.size() - 1; i >= 0; i--) {
            INDArray array = arrays.get(i);
            if (array != null && !array.wasClosed()) {
                array.close();
            }
        }
    }

    private int maskedArgmax(INDArray headOutput, int row, List<Integer> validIndices) {
        double maxQ = -Double.MAX_VALUE;
        int width = (int) headOutput.size(1);
        int best = firstInBounds(validIndices, width);
        for (int idx : validIndices) {
            if (idx < 0 || idx >= width) {
                continue;
            }
            double q = headOutput.getDouble(row, idx);
            if (q > maxQ) {
                maxQ = q;
                best = idx;
            }
        }
        return best;
    }

    private int firstInBounds(List<Integer> indices, int width) {
        for (int idx : indices) {
            if (idx >= 0 && idx < width) {
                return idx;
            }
        }
        return 0;
    }
}
