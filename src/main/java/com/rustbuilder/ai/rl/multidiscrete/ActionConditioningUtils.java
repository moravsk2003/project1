package com.rustbuilder.ai.rl.multidiscrete;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ActionConditioningUtils {

    /**
     * Creates a one-hot vector for a single sample with shape [1, size].
     */
    public static INDArray oneHot(int index, int size) {
        INDArray arr = Nd4j.zeros(1, size);
        arr.putScalar(0, index, 1.0);
        return arr;
    }

    /**
     * Creates a batch of one-hot vectors with shape [m, size].
     */
    public static INDArray oneHotBatch(int[] indices, int size) {
        int m = indices.length;
        INDArray arr = Nd4j.zeros(m, size);
        for (int i = 0; i < m; i++) {
            arr.putScalar(new int[]{i, indices[i]}, 1.0);
        }
        return arr;
    }

    /**
     * Concatenates baseFeatures and any extra arrays along dimension 1 (features).
     * Works for both single samples [1, features] and batches [m, features].
     */
    public static INDArray concat(INDArray baseFeatures, INDArray... extra) {
        if (extra == null || extra.length == 0) {
            return baseFeatures;
        }
        INDArray[] all = new INDArray[extra.length + 1];
        all[0] = baseFeatures;
        System.arraycopy(extra, 0, all, 1, extra.length);
        return Nd4j.concat(1, all);
    }
}
