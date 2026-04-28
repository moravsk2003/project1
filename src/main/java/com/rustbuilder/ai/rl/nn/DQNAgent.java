package com.rustbuilder.ai.rl.nn;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution3D;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.Cnn3DToFeedForwardPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;

public class DQNAgent {

    private MultiLayerNetwork mainNet;
    private MultiLayerNetwork targetNet;

    private int stateChannels;
    private int stateDepth;
    private int stateHeight;
    private int stateWidth;
    private int numActions;
    
    private double gamma = 0.99;

    public DQNAgent(int channels, int depth, int height, int width, int numActions) {
        this.stateChannels = channels;
        this.stateDepth = depth;
        this.stateHeight = height;
        this.stateWidth = width;
        this.numActions = numActions;
        
        mainNet = new MultiLayerNetwork(buildConfig());
        mainNet.init();
        
        targetNet = new MultiLayerNetwork(buildConfig());
        targetNet.init();
        updateTargetNetwork();
    }

    private MultiLayerConfiguration buildConfig() {
        // After Conv3D layer 0 (stride=1, pad=1): [batch, 16, 8, 8, 8]
        // After Conv3D layer 1 (stride=2, pad=1): [batch, 32, 4, 4, 4]
        // Flatten for Dense: 32 * 4 * 4 * 4 = 2048
        
        return new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .list()
                .layer(0, new Convolution3D.Builder(3, 3, 3)
                        .nIn(stateChannels)
                        .nOut(16)
                        .stride(1, 1, 1)
                        .padding(1, 1, 1)
                        .dataFormat(Convolution3D.DataFormat.NCDHW)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new Convolution3D.Builder(3, 3, 3)
                        .nIn(16)
                        .nOut(32)
                        .stride(2, 2, 2)
                        .padding(1, 1, 1)
                        .dataFormat(Convolution3D.DataFormat.NCDHW)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(32 * 4 * 4 * 4)  // Explicitly set flattened input size
                        .nOut(1024)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new DenseLayer.Builder()
                        .nIn(1024)
                        .nOut(1024)
                        .activation(Activation.RELU)
                        .build())
                .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(1024)
                        .nOut(numActions)
                        .activation(Activation.IDENTITY)
                        .build())
                .inputPreProcessor(2, new Cnn3DToFeedForwardPreProcessor(4, 4, 4, 32, true))
                .setInputType(InputType.convolutional3D(Convolution3D.DataFormat.NCDHW, stateDepth, stateHeight, stateWidth, stateChannels))
                .build();
    }

    public INDArray predict(INDArray state) {
        return mainNet.output(state, false);
    }
    
    public double[] getQValues(INDArray state) {
        try (INDArray output = predict(state)) {
            return output.toDoubleVector();
        }
    }

    public void updateTargetNetwork() {
        targetNet.setParams(mainNet.params().dup());
    }

    public double trainBatch(List<ExperienceReplay.Transition> batch) {
        int m = batch.size();
        
        INDArray[] statesArr = new INDArray[m];
        INDArray[] nextStatesArr = new INDArray[m];
        
        for (int i = 0; i < m; i++) {
            statesArr[i] = batch.get(i).state;
            nextStatesArr[i] = batch.get(i).nextState;
        }
        
        INDArray statesObj = Nd4j.concat(0, statesArr);
        INDArray nextStatesObj = Nd4j.concat(0, nextStatesArr);
        
        INDArray currentQs = mainNet.output(statesObj, false);
        INDArray targetQs = currentQs.dup();
        INDArray nextQs = targetNet.output(nextStatesObj, false);
        
        for (int i = 0; i < m; i++) {
            ExperienceReplay.Transition t = batch.get(i);
            
            double maxNextQ = 0.0;
            if (!t.isDone) {
                maxNextQ = -Double.MAX_VALUE;
                if (t.validNextActions != null && !t.validNextActions.isEmpty()) {
                    INDArray nextQRow = nextQs.getRow(i);
                    for (int a : t.validNextActions) {
                        if (a >= 0 && a < numActions) {
                            double q = nextQRow.getDouble(a);
                            if (q > maxNextQ) {
                                maxNextQ = q;
                            }
                        }
                    }
                } else {
                    maxNextQ = nextQs.getRow(i).maxNumber().doubleValue();
                }
            }
            
            double targetQ = t.reward + gamma * maxNextQ;
            
            if (t.action >= 0 && t.action < numActions) {
                 targetQs.putScalar(new int[]{i, t.action}, targetQ);
            }
        }
        
        mainNet.fit(statesObj, targetQs);
        double score = mainNet.score();
        
        if (statesObj != null && !statesObj.wasClosed()) statesObj.close();
        if (nextStatesObj != null && !nextStatesObj.wasClosed()) nextStatesObj.close();
        if (currentQs != null && !currentQs.wasClosed()) currentQs.close();
        if (targetQs != null && !targetQs.wasClosed()) targetQs.close();
        if (nextQs != null && !nextQs.wasClosed()) nextQs.close();
        
        return score;
    }

    public void save(String filepath) throws java.io.IOException {
        ModelSerializer.writeModel(mainNet, filepath, true);
    }

    public void load(String filepath) throws java.io.IOException {
        mainNet = ModelSerializer.restoreMultiLayerNetwork(filepath);
        if (targetNet != null) {
            targetNet.setParams(mainNet.params().dup());
        }
    }
}
