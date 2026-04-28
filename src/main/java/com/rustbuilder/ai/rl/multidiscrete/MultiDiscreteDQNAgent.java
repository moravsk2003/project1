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
import java.util.List;
import com.rustbuilder.ai.rl.RLRewardConfig;

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

    public MultiDiscreteDQNAgent(int channels, int depth, int height, int width) {
        this.stateChannels = channels;
        this.stateDepth = depth;
        this.stateHeight = height;
        this.stateWidth = width;
        
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

        return new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .graphBuilder()
                .addInputs("input", "cond_type", "cond_floor", "cond_tile", "cond_rot")
                .setInputTypes(
                        InputType.convolutional3D(Convolution3D.DataFormat.NCDHW, stateDepth, stateHeight, stateWidth, stateChannels),
                        InputType.feedForward(MultiDiscreteActionSpace.TYPE_COUNT),
                        InputType.feedForward(MultiDiscreteActionSpace.FLOOR_COUNT),
                        InputType.feedForward(MultiDiscreteActionSpace.TILE_COUNT),
                        InputType.feedForward(MultiDiscreteActionSpace.ROTATION_COUNT)
                )
                
                // Shared 3D CNN Trunk
                .addLayer("conv1", new Convolution3D.Builder(3, 3, 3)
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
                
                // Flatten and Shared Dense Block
                .inputPreProcessor("dense_shared1", new Cnn3DToFeedForwardPreProcessor(conv2Depth, conv2Height, conv2Width, 32, true))
                .addLayer("dense_shared1", new DenseLayer.Builder()
                        .nIn(32 * conv2Depth * conv2Height * conv2Width)
                        .nOut(1024)
                        .activation(Activation.RELU)
                        .build(), "conv2")
                .addLayer("dense_shared2", new DenseLayer.Builder()
                        .nIn(1024)
                        .nOut(1024)
                        .activation(Activation.RELU)
                        .build(), "dense_shared1")
                
                // Branching Action Heads (Conditional)
                .addLayer("out_type", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(1024).nOut(MultiDiscreteActionSpace.TYPE_COUNT).activation(Activation.IDENTITY).build(), "dense_shared2")
                
                .addVertex("concat_floor", new org.deeplearning4j.nn.conf.graph.MergeVertex(), "dense_shared2", "cond_type")
                .addLayer("out_floor", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(1024 + MultiDiscreteActionSpace.TYPE_COUNT).nOut(MultiDiscreteActionSpace.FLOOR_COUNT).activation(Activation.IDENTITY).build(), "concat_floor")
                
                .addVertex("concat_tile", new org.deeplearning4j.nn.conf.graph.MergeVertex(), "dense_shared2", "cond_type", "cond_floor")
                .addLayer("out_tile", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(1024 + MultiDiscreteActionSpace.TYPE_COUNT + MultiDiscreteActionSpace.FLOOR_COUNT).nOut(MultiDiscreteActionSpace.TILE_COUNT).activation(Activation.IDENTITY).build(), "concat_tile")
                
                .addVertex("concat_rot", new org.deeplearning4j.nn.conf.graph.MergeVertex(), "dense_shared2", "cond_type", "cond_floor", "cond_tile")
                .addLayer("out_rot", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(1024 + MultiDiscreteActionSpace.TYPE_COUNT + MultiDiscreteActionSpace.FLOOR_COUNT + MultiDiscreteActionSpace.TILE_COUNT).nOut(MultiDiscreteActionSpace.ROTATION_COUNT).activation(Activation.IDENTITY).build(), "concat_rot")
                
                .addVertex("concat_aimSector", new org.deeplearning4j.nn.conf.graph.MergeVertex(), "dense_shared2", "cond_type", "cond_floor", "cond_tile", "cond_rot")
                .addLayer("out_aimSector", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(1024 + MultiDiscreteActionSpace.TYPE_COUNT + MultiDiscreteActionSpace.FLOOR_COUNT + MultiDiscreteActionSpace.TILE_COUNT + MultiDiscreteActionSpace.ROTATION_COUNT).nOut(MultiDiscreteActionSpace.AIM_SECTOR_COUNT).activation(Activation.IDENTITY).build(), "concat_aimSector")
                
                .setOutputs("out_type", "out_floor", "out_tile", "out_rot", "out_aimSector")
                .build();
    }

    public INDArray encodeStateFeatures(INDArray stateBatch) {
        int m = (int) stateBatch.size(0);
        INDArray[] dummyInputs = new INDArray[5];
        dummyInputs[0] = stateBatch;
        dummyInputs[1] = Nd4j.zeros(m, MultiDiscreteActionSpace.TYPE_COUNT);
        dummyInputs[2] = Nd4j.zeros(m, MultiDiscreteActionSpace.FLOOR_COUNT);
        dummyInputs[3] = Nd4j.zeros(m, MultiDiscreteActionSpace.TILE_COUNT);
        dummyInputs[4] = Nd4j.zeros(m, MultiDiscreteActionSpace.ROTATION_COUNT);
        return mainNet.feedForward(dummyInputs, false).get("dense_shared2");
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
        
        INDArray[] statesArr = new INDArray[m];
        INDArray[] nextStatesArr = new INDArray[m];
        
        // Record taken actions for chosen paths
        int[] batchActType = new int[m];
        int[] batchActFloor = new int[m];
        int[] batchActTile = new int[m];
        int[] batchActRot = new int[m];
        int[] batchActAim = new int[m];
        
        for (int i = 0; i < m; i++) {
            MultiDiscreteExperienceReplay.Transition t = batch.get(i);
            statesArr[i] = t.state.dup();
            nextStatesArr[i] = t.nextState.dup();
            
            batchActType[i] = t.action.getTypeIndex();
            batchActFloor[i] = t.action.getFloorIndex();
            batchActTile[i] = t.action.getTileIndex();
            batchActRot[i] = t.action.getRotationIndex();
            batchActAim[i] = t.action.getAimSector();
        }
        
        INDArray statesObj = Nd4j.concat(0, statesArr);
        INDArray nextStatesObj = Nd4j.concat(0, nextStatesArr);
        
        // --- 1. Compute current Q-values with chosen action contexts ---
        INDArray condTypeObj = ActionConditioningUtils.oneHotBatch(batchActType, MultiDiscreteActionSpace.TYPE_COUNT);
        INDArray condFloorObj = ActionConditioningUtils.oneHotBatch(batchActFloor, MultiDiscreteActionSpace.FLOOR_COUNT);
        INDArray condTileObj = ActionConditioningUtils.oneHotBatch(batchActTile, MultiDiscreteActionSpace.TILE_COUNT);
        INDArray condRotObj = ActionConditioningUtils.oneHotBatch(batchActRot, MultiDiscreteActionSpace.ROTATION_COUNT);
        INDArray condAimObj = ActionConditioningUtils.oneHotBatch(batchActAim, MultiDiscreteActionSpace.AIM_SECTOR_COUNT);
        INDArray[] currentInputs = new INDArray[]{statesObj, condTypeObj, condFloorObj, condTileObj, condRotObj};
        
        INDArray[] currentQsList = mainNet.output(false, currentInputs);
        
        // Prepare target arrays
        INDArray[] targetQsList = new INDArray[5];
        for (int h = 0; h < 5; h++) {
            targetQsList[h] = currentQsList[h].dup();
        }
        
        // --- 2. Sequential Bootstrapping for Double DQN (Online Net for selection) ---
        INDArray onlineNextStateFeatures = encodeStateFeatures(nextStatesObj);
        INDArray onlineNextTypeLogits = predictType(onlineNextStateFeatures);
        
        boolean[] isDeadBranch = new boolean[m];
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
                // Determine valid types for next state
                List<Integer> vTypes = HeuristicMaskingUtils.getFeasibleTypes(t.nextGrid, hasTC, hasLootRoom, t.step + 1);
                if (vTypes.isEmpty()) {
                    isDeadBranch[i] = true;
                } else {
                    bestNextType[i] = getMaskedArgmax(onlineNextTypeLogits, i, vTypes);
                }
            }
        }
        
        INDArray nextCondTypeOnline = ActionConditioningUtils.oneHotBatch(bestNextType, MultiDiscreteActionSpace.TYPE_COUNT);
        INDArray onlineNextFloorLogits = predictFloor(ActionConditioningUtils.concat(onlineNextStateFeatures, nextCondTypeOnline));
        
        int[] bestNextFloor = new int[m];
        for (int i = 0; i < m; i++) {
            MultiDiscreteExperienceReplay.Transition t = batch.get(i);
            if (!t.isDone && t.nextGrid != null && !isDeadBranch[i]) {
                List<Integer> vFloors = HeuristicMaskingUtils.getValidFloors(t.nextGrid, bestNextType[i], t.step + 1);
                if (vFloors.isEmpty()) {
                    isDeadBranch[i] = true;
                } else {
                    bestNextFloor[i] = getMaskedArgmax(onlineNextFloorLogits, i, vFloors);
                }
            }
        }
        
        INDArray nextCondFloorOnline = ActionConditioningUtils.oneHotBatch(bestNextFloor, MultiDiscreteActionSpace.FLOOR_COUNT);
        INDArray onlineNextTileLogits = predictTile(ActionConditioningUtils.concat(onlineNextStateFeatures, nextCondTypeOnline, nextCondFloorOnline));
        
        int[] bestNextTile = new int[m];
        for (int i = 0; i < m; i++) {
            MultiDiscreteExperienceReplay.Transition t = batch.get(i);
            if (!t.isDone && t.nextGrid != null && !isDeadBranch[i]) {
                List<Integer> vTiles = HeuristicMaskingUtils.getValidTiles(t.nextGrid, bestNextType[i], bestNextFloor[i], t.step + 1);
                if (vTiles.isEmpty()) {
                    isDeadBranch[i] = true;
                } else {
                    bestNextTile[i] = getMaskedArgmax(onlineNextTileLogits, i, vTiles);
                }
            }
        }
        
        INDArray nextCondTileOnline = ActionConditioningUtils.oneHotBatch(bestNextTile, MultiDiscreteActionSpace.TILE_COUNT);
        INDArray onlineNextRotLogits = predictRot(ActionConditioningUtils.concat(onlineNextStateFeatures, nextCondTypeOnline, nextCondFloorOnline, nextCondTileOnline));
        
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
        INDArray nextCondRotOnline = ActionConditioningUtils.oneHotBatch(bestNextRot, MultiDiscreteActionSpace.ROTATION_COUNT);
        INDArray onlineNextAimLogits = predictAim(ActionConditioningUtils.concat(onlineNextStateFeatures, nextCondTypeOnline, nextCondFloorOnline, nextCondTileOnline, nextCondRotOnline));
        
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
        
        INDArray nextCondAimOnline = ActionConditioningUtils.oneHotBatch(bestNextAim, MultiDiscreteActionSpace.AIM_SECTOR_COUNT);
        INDArray[] targetNetNextQsList = targetNet.output(false, new INDArray[]{nextStatesObj, nextCondTypeOnline, nextCondFloorOnline, nextCondTileOnline, nextCondRotOnline});
        
        for (int i = 0; i < m; i++) {
            MultiDiscreteExperienceReplay.Transition t = batch.get(i);
            
            // HEAD-AWARE CREDIT ASSIGNMENT
            double r = t.reward;
            double typeR = r;
            double floorR = r;
            double tileR = r;
            double rotR = r;

            // Heuristic blame: if reward is severely negative, it's likely a mechanical failure
            if (r <= rewardConfig.penaltyNoSupport) {
                // Rotation/Floor are biggest suspects for NO_SUPPORT
                rotR *= 1.5;
                floorR *= 1.2;
            }
            if (r <= rewardConfig.penaltyBadSocket) {
                // Tile/Type are biggest suspects for BAD_SOCKET
                tileR *= 1.5;
                typeR *= 1.2;
            }

            for (int h = 0; h < 5; h++) {
                double baseR = switch(h) {
                    case 0 -> typeR;
                    case 1 -> floorR;
                    case 2 -> tileR;
                    case 3 -> rotR;
                    case 4 -> r; // aim (final step)
                    default -> 0.0;
                };

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
        
        // Cleanup arrays
        if (statesObj != null && !statesObj.wasClosed()) statesObj.close();
        if (nextStatesObj != null && !nextStatesObj.wasClosed()) nextStatesObj.close();
        if (condTypeObj != null && !condTypeObj.wasClosed()) condTypeObj.close();
        if (condFloorObj != null && !condFloorObj.wasClosed()) condFloorObj.close();
        if (condTileObj != null && !condTileObj.wasClosed()) condTileObj.close();
        if (condRotObj != null && !condRotObj.wasClosed()) condRotObj.close();
        if (nextCondTypeOnline != null && !nextCondTypeOnline.wasClosed()) nextCondTypeOnline.close();
        if (nextCondFloorOnline != null && !nextCondFloorOnline.wasClosed()) nextCondFloorOnline.close();
        if (nextCondTileOnline != null && !nextCondTileOnline.wasClosed()) nextCondTileOnline.close();
        if (nextCondRotOnline != null && !nextCondRotOnline.wasClosed()) nextCondRotOnline.close();
        if (nextCondAimOnline != null && !nextCondAimOnline.wasClosed()) nextCondAimOnline.close();
        if (condAimObj != null && !condAimObj.wasClosed()) condAimObj.close();
        
        for (int h = 0; h < 5; h++) {
            if (currentQsList[h] != null && !currentQsList[h].wasClosed()) currentQsList[h].close();
            if (targetQsList[h] != null && !targetQsList[h].wasClosed()) targetQsList[h].close();
            if (targetNetNextQsList[h] != null && !targetNetNextQsList[h].wasClosed()) targetNetNextQsList[h].close();
        }
        
        return score;
    }
    
    private int getMaskedArgmax(INDArray headOutput, int row, java.util.List<Integer> validIndices) {
        double maxQ = -Double.MAX_VALUE;
        int best = validIndices.get(0);
        for (int idx : validIndices) {
            double q = headOutput.getDouble(row, idx);
            if (q > maxQ) {
                maxQ = q;
                best = idx;
            }
        }
        return best;
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
