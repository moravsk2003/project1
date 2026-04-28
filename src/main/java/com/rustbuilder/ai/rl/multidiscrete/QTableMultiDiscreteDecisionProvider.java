package com.rustbuilder.ai.rl.multidiscrete;

import com.rustbuilder.ai.rl.legacy.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.config.GameConstants;

/**
 * [RL REDESIGN]
 * A Q-Learning based decision provider for the multi-discrete phase policy.
 * 
 * NOTE: Current implementation is PLACEMENT-ONLY for the experimental/learning path.
 * Tabular provider currently functions as a 5-phase provider inside the 5-phase contract
 * (aimSector is hardcoded to center=12 and not learned) to prevent state space explosion.
 * Stop semantics for multi-discrete mode are not yet formally defined, so the 
 * STOP action is excluded from candidate generation to prevent premature rollout termination 
 * and ensure the training loop only considers constructive actions.
 */
public class QTableMultiDiscreteDecisionProvider implements MultiDiscretePhaseDecisionProvider {

    // Phase dimensions for the tabular space
    private static final int TYPES = MultiDiscreteActionSpace.TYPE_COUNT; // 12
    private static final int FLOORS = MultiDiscreteActionSpace.FLOOR_COUNT; // 8
    private static final int TILES = MultiDiscreteActionSpace.TILE_COUNT; // 64
    private static final int ROTATIONS = MultiDiscreteActionSpace.ROTATION_COUNT; // 4
    
    // Total encoded actions: 12 * 8 * 64 * 4 = 24,576
    private static final int TOTAL_ACTION_SLOTS = TYPES * FLOORS * TILES * ROTATIONS;

    private final QTable qTable;
    private final Random random;
    
    // Learning parameters
    private double alpha = 0.1;
    private double gamma = 0.9;
    private double epsilon = 0.1;
    
    // Alignment constants — centralised in GameConstants
    private static final double GRID_ORIGIN_X = GameConstants.GRID_ORIGIN_X;
    private static final double GRID_ORIGIN_Y = GameConstants.GRID_ORIGIN_Y;

    public QTableMultiDiscreteDecisionProvider() {
        this.qTable = new QTable(TOTAL_ACTION_SLOTS);
        this.random = new Random();
    }

    public QTableMultiDiscreteDecisionProvider(QTable qTable) {
        this.qTable = qTable;
        this.random = new Random();
    }

    @Override
    public MultiDiscretePhaseDecision provideDecision(MultiDiscretePhaseContext context) {
        long state = computeStateHash(context);
        
        List<Integer> candidates = generateCandidateIndices(context);
        
        // Strategy: choose among candidate phase combinations
        int chosenId = qTable.selectAction(state, candidates, epsilon);
        
        return decodeActionId(chosenId);
    }

    /**
     * Performs a Q-value update based on the reward received.
     */
    public void update(MultiDiscretePhaseContext current, MultiDiscreteAction action, double reward, MultiDiscretePhaseContext next) {
        long s = computeStateHash(current);
        long sNext = computeStateHash(next);
        
        int aIdx = encodeActionId(action.getTypeIndex(), action.getFloorIndex(), 
                                  action.getTileIndex(), action.getRotationIndex());
        
        qTable.updateQValue(s, aIdx, reward, sNext, alpha, gamma);
    }

    // --- Type-Aware Candidate Generation ---

    private List<Integer> generateCandidateIndices(MultiDiscretePhaseContext context) {
        List<Integer> candidates = new ArrayList<>();
        
        // 1. Get valid types for current state
        List<Integer> validTypes = ActionSpace.getValidTypeActions(context.isHasTC(), context.isHasLootRoom(), context.getStep());
        
        for (int typeIdx : validTypes) {
            // [STOP Semantics Fix]
            // Skip STOP for multi-discrete learning path until separate stop logic is designed.
            if (typeIdx == ActionSpace.STOP_TYPE_INDEX) {
                continue; 
            }

            // [HEURISTIC GUIDANCE]
            // Get sensible subsets based on component type to keep the tabular search space relevant
            List<Integer> typeFloors = getCandidateFloorsForType(context, typeIdx);
            int[] typeRotations = getCandidateRotationsForType(typeIdx);

            for (int floor : typeFloors) {
                // Refine tiles based on the specific floor
                List<Integer> refinedTiles = HeuristicMaskingUtils.getValidTiles(context.getGrid(), typeIdx, floor, context.getStep());
                for (int tileIdx : refinedTiles) {
                    for (int rot : typeRotations) {
                        candidates.add(encodeActionId(typeIdx, floor, tileIdx, rot));
                    }
                }
            }
        }
        
        // Fallback: Safe central foundation placement instead of STOP
        if (candidates.isEmpty()) {
            candidates.add(encodeActionId(0, 0, 36, 0)); // Index 0 is Foundation, 36 is center
        }
        
        return candidates;
    }

    private List<Integer> getCandidateFloorsForType(MultiDiscretePhaseContext context, int typeIdx) {
        return HeuristicMaskingUtils.getValidFloors(
            context.getGrid(),
            typeIdx,
            context.getStep()
        );
    }


    private int[] getCandidateRotationsForType(int typeIdx) {
        com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType type = ActionSpace.decodeType(typeIdx);
        if (type == null) return new int[]{0};

        // For wall-like structures, use cardinal buckets to prune search space noise
        if (type == com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.WALL || 
            type == com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.DOORWAY || 
            type == com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.WINDOW_FRAME) {
            return new int[]{0, 1, 2, 3};
        }
        // Others: allow all rotations
        int[] allRotations = new int[MultiDiscreteActionSpace.ROTATION_COUNT];
        for (int i = 0; i < MultiDiscreteActionSpace.ROTATION_COUNT; i++) allRotations[i] = i;
        return allRotations;
    }

    // --- State & Action Mappings ---

    private long computeStateHash(MultiDiscretePhaseContext context) {
        long h = 0;
        List<BuildingBlock> blocks = new ArrayList<>(context.getGrid().getAllBlocks());
        
        // Stabilize hash by sorting blocks
        blocks.sort((b1, b2) -> {
            if (b1.getZ() != b2.getZ()) return Integer.compare(b1.getZ(), b2.getZ());
            int tx1 = (int) Math.round((b1.getX() - GRID_ORIGIN_X) / GameConstants.TILE_SIZE);
            int tx2 = (int) Math.round((b2.getX() - GRID_ORIGIN_X) / GameConstants.TILE_SIZE);
            if (tx1 != tx2) return Integer.compare(tx1, tx2);
            int ty1 = (int) Math.round((b1.getY() - GRID_ORIGIN_Y) / GameConstants.TILE_SIZE);
            int ty2 = (int) Math.round((b2.getY() - GRID_ORIGIN_Y) / GameConstants.TILE_SIZE);
            if (ty1 != ty2) return Integer.compare(ty1, ty2);
            return b1.getType().name().compareTo(b2.getType().name());
        });

        for (BuildingBlock b : blocks) {
            int tx = (int) Math.round((b.getX() - GRID_ORIGIN_X) / GameConstants.TILE_SIZE);
            int ty = (int) Math.round((b.getY() - GRID_ORIGIN_Y) / GameConstants.TILE_SIZE);
            int tz = b.getZ();
            if (tx >= 0 && tx < 8 && ty >= 0 && ty < 8) {
                // Combine type, coords into a unique value per block and mix it into the hash
                long blockBits = ((long)b.getType().ordinal() << 20) | (tx << 12) | (ty << 6) | tz;
                h = h * 31 + blockBits;
            }
        }
        h ^= (context.isHasTC() ? 0x5555555555555555L : 0);
        h ^= (context.isHasLootRoom() ? 0xAAAAAAAAAAAAAAAAL : 0);
        // Step removed from state hash to restore Markov property and enable generalization.
        return h;
    }

    private int encodeActionId(int type, int floor, int tileIdx, int rotIdx) {
        int id = type;
        id = id * FLOORS + floor;
        id = id * TILES + tileIdx;
        id = id * ROTATIONS + rotIdx;
        return id;
    }

    private MultiDiscretePhaseDecision decodeActionId(int id) {
        int rotIdx = id % ROTATIONS; id /= ROTATIONS;
        int tileIdx = id % TILES; id /= TILES;
        int floor = id % FLOORS; id /= FLOORS;
        int type = id % TYPES;
        
        // Увага: aimSector зафіксовано на 12 (центр тайлу). 
        // Додавання aimSector у Q-Table розширило б простір станів до 563,200 дій, 
        // що призведе до OutOfMemoryError. Для мікропозиціонування використовується виключно нейромережа.
        int fixedAimSector = 12;
        
        return new MultiDiscretePhaseDecision(type, floor, tileIdx, rotIdx, fixedAimSector);
    }

    public void setAlpha(double alpha) { this.alpha = alpha; }
    public void setGamma(double gamma) { this.gamma = gamma; }
    public void setEpsilon(double epsilon) { this.epsilon = epsilon; }
    public QTable getQTable() { return qTable; }
}
