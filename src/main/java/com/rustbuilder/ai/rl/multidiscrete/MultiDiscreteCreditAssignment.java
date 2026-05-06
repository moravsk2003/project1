package com.rustbuilder.ai.rl.multidiscrete;

import com.rustbuilder.core.placement.PlacementError;

/**
 * Assigns invalid-action blame only to the action heads that could have caused
 * the observed placement failure.
 */
public final class MultiDiscreteCreditAssignment {
    public static final int TYPE = 0;
    public static final int FLOOR = 1;
    public static final int TILE = 2;
    public static final int ROTATION = 3;
    public static final int AIM = 4;

    private static final double[] ALL_HEADS = {1.0, 1.0, 1.0, 1.0, 1.0};

    private MultiDiscreteCreditAssignment() {
    }

    public static double[] forPlacement(boolean success, PlacementError error) {
        if (success || error == null || error == PlacementError.NONE) {
            return ALL_HEADS.clone();
        }

        double[] mask = new double[MultiDiscreteActionSpace.PHASE_COUNT];
        switch (error) {
            case BAD_SOCKET_IS_FIRST:
            case GLOBAL_LIMIT:
                mark(mask, TYPE);
                break;
            case FLOOR_CONSTRAINT:
                mark(mask, TYPE, FLOOR);
                break;
            case BAD_SOCKET_NO_TARGET:
            case OUT_OF_BOUNDS:
                mark(mask, FLOOR, TILE, AIM);
                break;
            case BAD_SOCKET_WRONG_TARGET_TYPE:
                mark(mask, TYPE, FLOOR, TILE, AIM);
                break;
            case BAD_SOCKET_NO_SOCKET_ALIGNMENT:
            case BAD_SOCKET_CENTERDIST_REJECT:
                mark(mask, TILE, ROTATION, AIM);
                break;
            case NO_SUPPORT:
                mark(mask, FLOOR, TILE, ROTATION, AIM);
                break;
            case COLLISION:
                mark(mask, TYPE, FLOOR, TILE, ROTATION, AIM);
                break;
            case BAD_SOCKET:
            case UNKNOWN:
            default:
                mark(mask, TYPE, FLOOR, TILE, ROTATION, AIM);
                break;
        }
        return mask;
    }

    private static void mark(double[] mask, int... heads) {
        for (int head : heads) {
            if (head >= 0 && head < mask.length) {
                mask[head] = 1.0;
            }
        }
    }
}
