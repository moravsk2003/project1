package com.rustbuilder.ai.rl.multidiscrete;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution3D;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.Cnn3DToFeedForwardPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import com.rustbuilder.ai.rl.RLRewardConfig;
import com.rustbuilder.ai.rl.env.spec.StateEncodingSpec;
import com.rustbuilder.ai.rl.env.spec.ActionSpaceSpec;

public class MultiDiscreteDQNAgent {

    private ComputationGraph mainNet;
    private ComputationGraph targetNet;

    private int stateChannels;
    private int stateDepth;
    private int stateHeight;
    private int stateWidth;
    
    private double gamma = 0.99;
    private RLRewardConfig rewardConfig = RLRewardConfig.createDefault();

    public void setRewardConfig(RLRewardConfig rewardConfig) {
        if (rewardConfig != null) {
            this.rewardConfig = rewardConfig;
        }
    }

    private StateEncodingSpec stateSpec;
    private ActionSpaceSpec actionSpec;

    public MultiDiscreteDQNAgent(StateEncodingSpec stateSpec, ActionSpaceSpec actionSpec) {
        this.stateSpec = stateSpec;
        this.actionSpec = actionSpec;
        this.stateChannels = stateSpec.voxelChannels;
        this.stateDepth = stateSpec.gridShape[2];
        this.stateHeight = stateSpec.gridShape[1];
        this.stateWidth = stateSpec.gridShape[0];
        
        mainNet = new ComputationGraph(buildConfig());
        mainNet.init();
        
        targetNet = new ComputationGraph(buildConfig());
        targetNet.init();
        updateTargetNetwork();
    }

    private ComputationGraphConfiguration buildConfig() {
        int conv1Depth = stateDepth;
        int conv1Height = stateHeight;
        int conv1Width = stateWidth;
        
        int conv2Depth = (conv1Depth - 3 + 2 * 1) / 2 + 1;
        int conv2Height = (conv1Height - 3 + 2 * 1) / 2 + 1;
        int conv2Width = (conv1Width - 3 + 2 * 1) / 2 + 1;

        org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .graphBuilder();

        if (stateSpec.hasGlobalVector) {
            builder.addInputs("input", "input_global", "cond_type", "cond_floor", "cond_tile", "cond_rot")
                   .setInputTypes(
                           InputType.convolutional3D(Convolution3D.DataFormat.NCDHW, stateDepth, stateHeight, stateWidth, stateChannels),
                           InputType.feedForward(stateSpec.globalFeatureCount),
                           InputType.feedForward(actionSpec.typeCount),
                           InputType.feedForward(actionSpec.floorCount),
                           InputType.feedForward(actionSpec.tileCount),
                           InputType.feedForward(actionSpec.rotationCount)
                   );
        } else {
            builder.addInputs("input", "cond_type", "cond_floor", "cond_tile", "cond_rot")
                   .setInputTypes(
                           InputType.convolutional3D(Convolution3D.DataFormat.NCDHW, stateDepth, stateHeight, stateWidth, stateChannels),
                           InputType.feedForward(actionSpec.typeCount),
                           InputType.feedForward(actionSpec.floorCount),
                           InputType.feedForward(actionSpec.tileCount),
                           InputType.feedForward(actionSpec.rotationCount)
                   );
        }

        builder.addLayer("conv1", new Convolution3D.Builder(3, 3, 3)
                .nIn(stateChannels)
                .nOut(16)
                .stride(1, 1, 1)
                .padding(1, 1, 1)
                .dataFormat(Convolution3D.DataFormat.NCDHW)
                .activation(Activation.RELU)
                .build(), "input")
        .addLayer("conv2", new Convolution3D.Builder(3, 3, 3)
                .nIn(16)
                .nOut(32)
                .stride(2, 2, 2)
                .padding(1, 1, 1)
                .dataFormat(Convolution3D.DataFormat.NCDHW)
                .activation(Activation.RELU)
                .build(), "conv1")
        .inputPreProcessor("dense_shared1", new Cnn3DToFeedForwardPreProcessor(conv2Depth, conv2Height, conv2Width, 32, true))
        .addLayer("dense_shared1", new DenseLayer.Builder()
                .nIn(32 * conv2Depth * conv2Height * conv2Width)
                .nOut(1024)
                .activation(Activation.RELU)
                .build(), "conv2");

        String nextLayerInput = "dense_shared1";
        int nextLayerIn = 1024;

        if (stateSpec.hasGlobalVector) {
            builder.addLayer("global_dense", new DenseLayer.Builder()
                    .nIn(stateSpec.globalFeatureCount)
                    .nOut(128)
                    .activation(Activation.RELU)
                    .build(), "input_global");
            builder.addVertex("concat_embeddings", new org.deeplearning4j.nn.conf.graph.MergeVertex(), "dense_shared1", "global_dense");
            nextLayerInput = "concat_embeddings";
            nextLayerIn = 1024 + 128;
        }

        builder.addLayer("dense_shared2", new DenseLayer.Builder()
                .nIn(nextLayerIn)
                .nOut(1024)
                .activation(Activation.RELU)
                .build(), nextLayerInput);

        builder.addLayer("out_type", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .nIn(1024).nOut(actionSpec.typeCount).activation(Activation.IDENTITY).build(), "dense_shared2")
                
        .addVertex("concat_floor", new org.deeplearning4j.nn.conf.graph.MergeVertex(), "dense_shared2", "cond_type")
        .addLayer("out_floor", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .nIn(1024 + actionSpec.typeCount).nOut(actionSpec.floorCount).activation(Activation.IDENTITY).build(), "concat_floor")
                
        .addVertex("concat_tile", new org.deeplearning4j.nn.conf.graph.MergeVertex(), "dense_shared2", "cond_type", "cond_floor")
        .addLayer("out_tile", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .nIn(1024 + actionSpec.typeCount + actionSpec.floorCount).nOut(actionSpec.tileCount).activation(Activation.IDENTITY).build(), "concat_tile")
                
        .addVertex("concat_rot", new org.deeplearning4j.nn.conf.graph.MergeVertex(), "dense_shared2", "cond_type", "cond_floor", "cond_tile")
        .addLayer("out_rot", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .nIn(1024 + actionSpec.typeCount + actionSpec.floorCount + actionSpec.tileCount).nOut(actionSpec.rotationCount).activation(Activation.IDENTITY).build(), "concat_rot")
                
        .addVertex("concat_aimSector", new org.deeplearning4j.nn.conf.graph.MergeVertex(), "dense_shared2", "cond_type", "cond_floor", "cond_tile", "cond_rot")
        .addLayer("out_aimSector", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .nIn(1024 + actionSpec.typeCount + actionSpec.floorCount + actionSpec.tileCount + actionSpec.rotationCount).nOut(actionSpec.aimCount).activation(Activation.IDENTITY).build(), "concat_aimSector")
                
        .setOutputs("out_type", "out_floor", "out_tile", "out_rot", "out_aimSector");

        return builder.build();
    }

    public INDArray encodeStateFeatures(INDArray stateBatch, INDArray globalBatch) {
        int m = (int) stateBatch.size(0);
        if (stateSpec.hasGlobalVector) {
            INDArray[] dummyInputs = new INDArray[6];
            dummyInputs[0] = stateBatch;
            dummyInputs[1] = globalBatch;
            dummyInputs[2] = Nd4j.zeros(m, actionSpec.typeCount);
            dummyInputs[3] = Nd4j.zeros(m, actionSpec.floorCount);
            dummyInputs[4] = Nd4j.zeros(m, actionSpec.tileCount);
            dummyInputs[5] = Nd4j.zeros(m, actionSpec.rotationCount);
            return mainNet.feedForward(dummyInputs, false).get("dense_shared2");
        } else {
            INDArray[] dummyInputs = new INDArray[5];
            dummyInputs[0] = stateBatch;
            dummyInputs[1] = Nd4j.zeros(m, actionSpec.typeCount);
            dummyInputs[2] = Nd4j.zeros(m, actionSpec.floorCount);
            dummyInputs[3] = Nd4j.zeros(m, actionSpec.tileCount);
            dummyInputs[4] = Nd4j.zeros(m, actionSpec.rotationCount);
            return mainNet.feedForward(dummyInputs, false).get("dense_shared2");
        }
    }

    public INDArray predictType(INDArray stateFeatures) {
        return mainNet.getLayer("out_type").activate(stateFeatures, false, org.deeplearning4j.nn.workspace.LayerWorkspaceMgr.noWorkspaces());
    }

    public INDArray predictFloor(INDArray concatInput) {
        return mainNet.getLayer("out_floor").activate(concatInput, false, org.deeplearning4j.nn.workspace.LayerWorkspaceMgr.noWorkspaces());
    }

    public INDArray predictTile(INDArray concatInput) {
        return mainNet.getLayer("out_tile").activate(concatInput, false, org.deeplearning4j.nn.workspace.LayerWorkspaceMgr.noWorkspaces());
    }

    public INDArray predictRot(INDArray concatInput) {
        return mainNet.getLayer("out_rot").activate(concatInput, false, org.deeplearning4j.nn.workspace.LayerWorkspaceMgr.noWorkspaces());
    }

    public INDArray predictAim(INDArray concatInput) {
        return mainNet.getLayer("out_aimSector").activate(concatInput, false, org.deeplearning4j.nn.workspace.LayerWorkspaceMgr.noWorkspaces());
    }

    public void updateTargetNetwork() {
        targetNet.setParams(mainNet.params().dup());
    }

    public double trainBatch(List<MultiDiscreteExperienceReplay.Transition> batch) {
        int m = batch.size();
        if (m == 0) return 0.0;

        List<INDArray> ownedArrays = new ArrayList<>();
        try {
        INDArray[] statesArr = new INDArray[m];
        INDArray[] nextStatesArr = new INDArray[m];
        INDArray[] statesGlobalArr = stateSpec.hasGlobalVector ? new INDArray[m] : null;
        INDArray[] nextStatesGlobalArr = stateSpec.hasGlobalVector ? new INDArray[m] : null;
        
        // Record taken actions for chosen paths
        int[] batchActType = new int[m];
        int[] batchActFloor = new int[m];
        int[] batchActTile = new int[m];
        int[] batchActRot = new int[m];
        int[] batchActAim = new int[m];
        
        for (int i = 0; i < m; i++) {
            MultiDiscreteExperienceReplay.Transition t = batch.get(i);
            statesArr[i] = t.state.voxelTensor().dup();
            nextStatesArr[i] = t.nextState.voxelTensor().dup();
            ownedArrays.add(statesArr[i]);
            ownedArrays.add(nextStatesArr[i]);
            if (stateSpec.hasGlobalVector) {
                statesGlobalArr[i] = t.state.getGlobalVector().dup();
                nextStatesGlobalArr[i] = t.nextState.getGlobalVector().dup();
                ownedArrays.add(statesGlobalArr[i]);
                ownedArrays.add(nextStatesGlobalArr[i]);
            }
            
            batchActType[i] = t.action.getTypeIndex();
            batchActFloor[i] = t.action.getFloorIndex();
            batchActTile[i] = t.action.getTileIndex();
            batchActRot[i] = t.action.getRotationIndex();
            batchActAim[i] = t.action.getAimSector();
        }

        INDArray statesObj = Nd4j.concat(0, statesArr);
        INDArray nextStatesObj = Nd4j.concat(0, nextStatesArr);
        ownedArrays.add(statesObj);
        ownedArrays.add(nextStatesObj);
        
        // --- 1. Compute current Q-values with chosen action contexts ---
        INDArray condTypeObj = ActionConditioningUtils.oneHotBatch(batchActType, actionSpec.typeCount);
        INDArray condFloorObj = ActionConditioningUtils.oneHotBatch(batchActFloor, actionSpec.floorCount);
        INDArray condTileObj = ActionConditioningUtils.oneHotBatch(batchActTile, actionSpec.tileCount);
        INDArray condRotObj = ActionConditioningUtils.oneHotBatch(batchActRot, actionSpec.rotationCount);
        ownedArrays.add(condTypeObj);
        ownedArrays.add(condFloorObj);
        ownedArrays.add(condTileObj);
        ownedArrays.add(condRotObj);
        INDArray[] currentInputs;
        INDArray statesGlobalObj = stateSpec.hasGlobalVector ? Nd4j.concat(0, statesGlobalArr) : null;
        if (statesGlobalObj != null) ownedArrays.add(statesGlobalObj);
        if (stateSpec.hasGlobalVector) {
            currentInputs = new INDArray[]{statesObj, statesGlobalObj, condTypeObj, condFloorObj, condTileObj, condRotObj};
        } else {
            currentInputs = new INDArray[]{statesObj, condTypeObj, condFloorObj, condTileObj, condRotObj};
        }
        
        INDArray[] currentQsList = mainNet.output(false, currentInputs);
        for (INDArray currentQs : currentQsList) {
            ownedArrays.add(currentQs);
        }
        
        // Prepare target arrays
        INDArray[] targetQsList = new INDArray[5];
        for (int h = 0; h < 5; h++) {
            targetQsList[h] = currentQsList[h].dup();
            ownedArrays.add(targetQsList[h]);
        }
        
        // --- 2. Sequential Bootstrapping for Double DQN (Online Net for selection) ---
        INDArray nextStatesGlobalObj = stateSpec.hasGlobalVector ? Nd4j.concat(0, nextStatesGlobalArr) : null;
        INDArray onlineNextStateFeatures = encodeStateFeatures(nextStatesObj, nextStatesGlobalObj);
        INDArray onlineNextTypeLogits = predictType(onlineNextStateFeatures);
        if (nextStatesGlobalObj != null) ownedArrays.add(nextStatesGlobalObj);
        ownedArrays.add(onlineNextStateFeatures);
        ownedArrays.add(onlineNextTypeLogits);
        
        boolean[] isDeadBranch = new boolean[m];
        MultiDiscretePhaseContext[] nextContexts = new MultiDiscretePhaseContext[m];
        int[] bestNextType = new int[m];
        for (int i = 0; i < m; i++) {
            MultiDiscreteExperienceReplay.Transition t = batch.get(i);
            if (!t.isDone && t.nextGrid != null) {
                boolean hasTC = false;
                boolean hasLootRoom = false;
                for (com.rustbuilder.model.core.BuildingBlock b : t.nextGrid.getAllBlocks()) {
                    if (b.getType() == com.rustbuilder.model.core.BuildingType.TC) hasTC = true;
                        if (b.getType() == com.rustbuilder.model.core.BuildingType.LOOT_ROOM) hasLootRoom = true;
                }
                nextContexts[i] = new MultiDiscretePhaseContext(t.nextGrid, hasTC, hasLootRoom, t.step + 1, t.step + 2);
                // Determine valid types for next state
                List<Integer> vTypes = HeuristicMaskingUtils.getFeasibleTypes(nextContexts[i]);
                if (vTypes.isEmpty()) {
                    isDeadBranch[i] = true;
                } else {
                    bestNextType[i] = getMaskedArgmax(onlineNextTypeLogits, i, vTypes);
                }
            }
        }
        
        INDArray nextCondTypeOnline = ActionConditioningUtils.oneHotBatch(bestNextType, actionSpec.typeCount);
        INDArray nextFloorInput = ActionConditioningUtils.concat(onlineNextStateFeatures, nextCondTypeOnline);
        INDArray onlineNextFloorLogits = predictFloor(nextFloorInput);
        ownedArrays.add(nextCondTypeOnline);
        ownedArrays.add(nextFloorInput);
        ownedArrays.add(onlineNextFloorLogits);
        
        int[] bestNextFloor = new int[m];
        for (int i = 0; i < m; i++) {
            MultiDiscreteExperienceReplay.Transition t = batch.get(i);
            if (!t.isDone && t.nextGrid != null && !isDeadBranch[i]) {
                List<Integer> vFloors = HeuristicMaskingUtils.getValidFloors(nextContexts[i], bestNextType[i]);
                if (vFloors.isEmpty()) {
                    isDeadBranch[i] = true;
                } else {
                    bestNextFloor[i] = getMaskedArgmax(onlineNextFloorLogits, i, vFloors);
                }
            }
        }
        
        INDArray nextCondFloorOnline = ActionConditioningUtils.oneHotBatch(bestNextFloor, actionSpec.floorCount);
        INDArray nextTileInput = ActionConditioningUtils.concat(onlineNextStateFeatures, nextCondTypeOnline, nextCondFloorOnline);
        INDArray onlineNextTileLogits = predictTile(nextTileInput);
        ownedArrays.add(nextCondFloorOnline);
        ownedArrays.add(nextTileInput);
        ownedArrays.add(onlineNextTileLogits);
        
        int[] bestNextTile = new int[m];
        for (int i = 0; i < m; i++) {
            MultiDiscreteExperienceReplay.Transition t = batch.get(i);
            if (!t.isDone && t.nextGrid != null && !isDeadBranch[i]) {
                List<Integer> vTiles = HeuristicMaskingUtils.getValidTiles(nextContexts[i], bestNextType[i], bestNextFloor[i]);
                if (vTiles.isEmpty()) {
                    isDeadBranch[i] = true;
                } else {
                    bestNextTile[i] = getMaskedArgmax(onlineNextTileLogits, i, vTiles);
                }
            }
        }
        
        INDArray nextCondTileOnline = ActionConditioningUtils.oneHotBatch(bestNextTile, actionSpec.tileCount);
        INDArray nextRotInput = ActionConditioningUtils.concat(onlineNextStateFeatures, nextCondTypeOnline, nextCondFloorOnline, nextCondTileOnline);
        INDArray onlineNextRotLogits = predictRot(nextRotInput);
        ownedArrays.add(nextCondTileOnline);
        ownedArrays.add(nextRotInput);
        ownedArrays.add(onlineNextRotLogits);
        
        int[] bestNextRot = new int[m];
        for (int i = 0; i < m; i++) {
            MultiDiscreteExperienceReplay.Transition t = batch.get(i);
            if (!t.isDone && t.nextGrid != null && !isDeadBranch[i]) {
                List<Integer> vRots = HeuristicMaskingUtils.getValidRotations(t.nextGrid, bestNextType[i], bestNextFloor[i], bestNextTile[i]);
                if (vRots.isEmpty()) {
                    isDeadBranch[i] = true;
                } else {
                    bestNextRot[i] = getMaskedArgmax(onlineNextRotLogits, i, vRots);
                }
            }
        }

        // --- 3. Evaluate Bootstrapped actions with Target Net ---
        INDArray nextCondRotOnline = ActionConditioningUtils.oneHotBatch(bestNextRot, actionSpec.rotationCount);
        INDArray nextAimInput = ActionConditioningUtils.concat(onlineNextStateFeatures, nextCondTypeOnline, nextCondFloorOnline, nextCondTileOnline, nextCondRotOnline);
        INDArray onlineNextAimLogits = predictAim(nextAimInput);
        ownedArrays.add(nextCondRotOnline);
        ownedArrays.add(nextAimInput);
        ownedArrays.add(onlineNextAimLogits);
        
        int[] bestNextAim = new int[m];
        for (int i = 0; i < m; i++) {
            MultiDiscreteExperienceReplay.Transition t = batch.get(i);
            if (!t.isDone && t.nextGrid != null && !isDeadBranch[i]) {
                List<Integer> vAim = HeuristicMaskingUtils.getValidAimSectors(t.nextGrid, bestNextType[i], bestNextFloor[i], bestNextTile[i], bestNextRot[i]);
                if (vAim.isEmpty()) {
                    isDeadBranch[i] = true;
                } else {
                    bestNextAim[i] = getMaskedArgmax(onlineNextAimLogits, i, vAim);
                }
            }
        }
        
        INDArray[] targetInputs;
        if (stateSpec.hasGlobalVector) {
            targetInputs = new INDArray[]{nextStatesObj, nextStatesGlobalObj, nextCondTypeOnline, nextCondFloorOnline, nextCondTileOnline, nextCondRotOnline};
        } else {
            targetInputs = new INDArray[]{nextStatesObj, nextCondTypeOnline, nextCondFloorOnline, nextCondTileOnline, nextCondRotOnline};
        }
        INDArray[] targetNetNextQsList = targetNet.output(false, targetInputs);
        for (INDArray targetNetNextQs : targetNetNextQsList) {
            ownedArrays.add(targetNetNextQs);
        }
        
        for (int i = 0; i < m; i++) {
            MultiDiscreteExperienceReplay.Transition t = batch.get(i);
            
            double r = t.reward;
            double[] headMultipliers = t.headRewardMultipliers;

            for (int h = 0; h < 5; h++) {
                double multiplier = (headMultipliers != null && h < headMultipliers.length)
                    ? headMultipliers[h]
                    : 1.0;
                double baseR = r * multiplier;

                double targetQ_h = baseR;
                
                if (!t.isDone && t.nextGrid != null && !isDeadBranch[i]) {
                    int chosenAction_h = switch (h) {
                        case 0 -> bestNextType[i];
                        case 1 -> bestNextFloor[i];
                        case 2 -> bestNextTile[i];
                        case 3 -> bestNextRot[i];
                        case 4 -> bestNextAim[i];
                        default -> 0;
                    };
                    targetQ_h += gamma * targetNetNextQsList[h].getDouble(i, chosenAction_h);
                }
                
                int takenAction_h = switch (h) {
                    case 0 -> batchActType[i];
                    case 1 -> batchActFloor[i];
                    case 2 -> batchActTile[i];
                    case 3 -> batchActRot[i];
                    case 4 -> batchActAim[i];
                    default -> 0;
                };
                
                targetQsList[h].putScalar(new int[]{i, takenAction_h}, targetQ_h);
            }
        }
        
        mainNet.fit(currentInputs, targetQsList);
        double score = mainNet.score();
        return score;
        } finally {
            closeAll(ownedArrays);
        }
    }

    private void closeAll(List<INDArray> arrays) {
        for (int i = arrays.size() - 1; i >= 0; i--) {
            INDArray array = arrays.get(i);
            if (array != null && !array.wasClosed()) {
                array.close();
            }
        }
    }
    
    private int getMaskedArgmax(INDArray headOutput, int row, java.util.List<Integer> validIndices) {
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

    private int firstInBounds(java.util.List<Integer> indices, int width) {
        for (int idx : indices) {
            if (idx >= 0 && idx < width) {
                return idx;
            }
        }
        return 0;
    }

    public void save(String filepath) throws IOException {
        ModelSerializer.writeModel(mainNet, filepath, true);
    }

    public void load(String filepath) throws IOException {
        mainNet = ModelSerializer.restoreComputationGraph(filepath);
        if (targetNet != null) {
            targetNet.setParams(mainNet.params().dup());
        }
    }
}
