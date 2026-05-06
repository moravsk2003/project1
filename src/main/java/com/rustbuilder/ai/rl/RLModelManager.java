package com.rustbuilder.ai.rl;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteActionSpace;




/**
 * Manages RL model persistence — save/load neural networks and parameters.
 */
public class RLModelManager {

    private static final String MODELS_DIR = "models_rl";

    /**
     * Serializable snapshot of training state.
     */
    public static class RLModel implements Serializable {
        private static final long serialVersionUID = 1L;

        public final String name;
        public final int episodesTrained;
        public final double bestScore;
        public final double epsilon;
        public final double logisticsWeight;
        public final double costWeight;
        public final double raidWeight;
        public final double workingAreaWeight;
        public final double safeZoneWeight;
        public RLRewardConfig rewardConfig; 

        // Compatibility metadata
        public String stateEncoderName = "Voxel";
        public String stateEncoderVersion = "v1";
        public int voxelChannels = 11;
        public int gridWidth = 8;
        public int gridHeight = 8;
        public int gridFloors = 8;
        public int actionTypeCount = 11;
        public int actionFloorCount = 8;
        public int actionTileCount = 64;
        public int actionRotationCount = MultiDiscreteActionSpace.ROTATION_COUNT;
        public int actionAimCount = MultiDiscreteActionSpace.AIM_SECTOR_COUNT;
        public String tileIndexingMode = "LEGACY_64";
        public boolean hasGlobalVector = false;
        public int globalFeatureCount = 0;
        public boolean hasObjectTable = false;
        public boolean hasGraphState = false;

        public RLModel(String name, int episodesTrained,
                       double bestScore, double epsilon,
                       double logisticsWeight, double costWeight, double raidWeight, double workingAreaWeight,
                       double safeZoneWeight,
                       RLRewardConfig rewardConfig) {
            this.name = name;
            this.episodesTrained = episodesTrained;
            this.bestScore = bestScore;
            this.epsilon = epsilon;
            this.logisticsWeight = logisticsWeight;
            this.costWeight = costWeight;
            this.raidWeight = raidWeight;
            this.workingAreaWeight = workingAreaWeight;
            this.safeZoneWeight = safeZoneWeight;
            this.rewardConfig = rewardConfig;
        }

        @Override
        public String toString() {
            return String.format("%s (Ep %d, Best: %.3f)", name, episodesTrained, bestScore);
        }
    }

    private static Path getModelsDir() {
        Path dir = Paths.get(MODELS_DIR);
        try {
            Files.createDirectories(dir);
        } catch (IOException e) {
        }
        return dir;
    }

    public static void saveModel(RLModel model, RLTrainingService rlService) throws IOException {
        Path file = getModelsDir().resolve(model.name + ".rmeta");
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new BufferedOutputStream(Files.newOutputStream(file)))) {
            oos.writeObject(model);
        }
        Path netFile = getModelsDir().resolve(model.name + ".rnet");
        rlService.getMultiDiscreteAgent().save(netFile.toString());
    }

    public static RLModel loadModel(String name, RLTrainingService rlService) throws IOException, ClassNotFoundException {
        RLModel model = loadMetadata(name);
        loadNetworkWeights(name, rlService);
        return model;
    }

    public static RLModel loadMetadata(String name) throws IOException, ClassNotFoundException {
        Path file = getModelsDir().resolve(name + ".rmeta");
        try (ObjectInputStream ois = new ObjectInputStream(
                new BufferedInputStream(Files.newInputStream(file)))) {
            return (RLModel) ois.readObject();
        }
    }

    public static void loadNetworkWeights(String name, RLTrainingService rlService) throws IOException {
        Path netFile = getModelsDir().resolve(name + ".rnet");
        if (Files.exists(netFile)) {
            rlService.getMultiDiscreteAgent().load(netFile.toString());
        }
    }

    public static List<String> listModels() {
        Path dir = getModelsDir();
        try (Stream<Path> files = Files.list(dir)) {
            return files
                .filter(p -> p.toString().endsWith(".rmeta"))
                .map(p -> {
                    String fileName = p.getFileName().toString();
                    return fileName.substring(0, fileName.length() - 6); // remove .rmeta
                })
                .sorted()
                .collect(Collectors.toList());
        } catch (IOException e) {
            return new ArrayList<>();
        }
    }

    public static boolean deleteModel(String name) {
        Path meta = getModelsDir().resolve(name + ".rmeta");
        Path net = getModelsDir().resolve(name + ".rnet");
        try {
            boolean d1 = Files.deleteIfExists(meta);
            boolean d2 = Files.deleteIfExists(net);
            return d1 || d2;
        } catch (IOException e) {
            return false;
        }
    }

    public static RLModel createSnapshot(String name, RLTrainingService rlService,
                                        double logW, double costW, double raidW, double workingAreaW, double safeZoneW) {
        RLRewardConfig cfg = rlService.getRewardConfig();
        if (cfg == null) {
            cfg = RLRewardConfig.createDefault();
        }

        RLModel model = new RLModel(name, rlService.getEpisodesTrained(),
                           rlService.getBestScore(), rlService.getEpsilon(),
                           logW, costW, raidW, workingAreaW, safeZoneW,
                           cfg.clone());
                           
        com.rustbuilder.ai.rl.env.spec.EncodingRuntimeConfig config = rlService.getRuntimeConfig();
        model.stateEncoderName = config.stateEncodingSpec.encoderName;
        model.stateEncoderVersion = config.stateEncodingSpec.encoderVersion;
        model.voxelChannels = config.stateEncodingSpec.voxelChannels;
        model.gridWidth = config.gridSpec.width;
        model.gridHeight = config.gridSpec.height;
        model.gridFloors = config.gridSpec.floors;
        model.actionTypeCount = config.actionSpaceSpec.typeCount;
        model.actionFloorCount = config.actionSpaceSpec.floorCount;
        model.actionTileCount = config.actionSpaceSpec.tileCount;
        model.actionRotationCount = config.actionSpaceSpec.rotationCount;
        model.actionAimCount = config.actionSpaceSpec.aimCount;
        model.tileIndexingMode = config.actionSpaceSpec.tileIndexingMode.name();
        model.hasGlobalVector = config.stateEncodingSpec.hasGlobalVector;
        model.globalFeatureCount = config.stateEncodingSpec.globalFeatureCount;
        model.hasObjectTable = config.stateEncodingSpec.hasObjectTable;
        model.hasGraphState = config.stateEncodingSpec.hasGraphState;
        
        return model;
    }

    public static void restoreFromModel(RLTrainingService rlService, RLModel model) {
        com.rustbuilder.ai.rl.env.spec.EncodingRuntimeConfig currentConfig = rlService.getRuntimeConfig();
        
        // Strict model compatibility check
        if (!model.stateEncoderName.equals(currentConfig.stateEncodingSpec.encoderName) ||
            !model.stateEncoderVersion.equals(currentConfig.stateEncodingSpec.encoderVersion) ||
            model.voxelChannels != currentConfig.stateEncodingSpec.voxelChannels ||
            model.gridWidth != currentConfig.gridSpec.width ||
            model.gridHeight != currentConfig.gridSpec.height ||
            model.gridFloors != currentConfig.gridSpec.floors ||
            model.actionTypeCount != currentConfig.actionSpaceSpec.typeCount ||
            model.actionFloorCount != currentConfig.actionSpaceSpec.floorCount ||
            model.actionTileCount != currentConfig.actionSpaceSpec.tileCount ||
            model.actionRotationCount != currentConfig.actionSpaceSpec.rotationCount ||
            model.actionAimCount != currentConfig.actionSpaceSpec.aimCount ||
            !model.tileIndexingMode.equals(currentConfig.actionSpaceSpec.tileIndexingMode.name()) ||
            model.hasGlobalVector != currentConfig.stateEncodingSpec.hasGlobalVector ||
            model.globalFeatureCount != currentConfig.stateEncodingSpec.globalFeatureCount ||
            model.hasObjectTable != currentConfig.stateEncodingSpec.hasObjectTable ||
            model.hasGraphState != currentConfig.stateEncodingSpec.hasGraphState) {
            throw new IllegalArgumentException("Model compatibility check failed! " +
                    "Model uses encoder version " + model.stateEncoderVersion + 
                    " (" + model.voxelChannels + " channels), " +
                    "but current runtime is " + currentConfig.stateEncodingSpec.encoderVersion + 
                    " (" + currentConfig.stateEncodingSpec.voxelChannels + " channels).");
        }

        rlService.resetRuntimeState();
        rlService.setEpisodesTrained(model.episodesTrained);
        rlService.setEpsilon(model.epsilon);
        rlService.setBestScore(model.bestScore);
        if (model.rewardConfig != null) {
            rlService.setRewardConfig(model.rewardConfig.clone());
        }
    }
}
