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

import com.rustbuilder.ai.rl.legacy.QTable;
import com.rustbuilder.ai.rl.multidiscrete.QTableMultiDiscreteDecisionProvider;

/**
 * Manages RL model persistence — save/load neural networks, Q-Tables, and parameters.
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
        public RLRewardConfig rewardConfig; // Added for persistence

        public RLModel(String name, int episodesTrained,
                       double bestScore, double epsilon,
                       double logisticsWeight, double costWeight, double raidWeight, double workingAreaWeight,
                       RLRewardConfig rewardConfig) {
            this.name = name;
            this.episodesTrained = episodesTrained;
            this.bestScore = bestScore;
            this.epsilon = epsilon;
            this.logisticsWeight = logisticsWeight;
            this.costWeight = costWeight;
            this.raidWeight = raidWeight;
            this.workingAreaWeight = workingAreaWeight;
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
        rlService.getAgent().save(netFile.toString());
        
        // Save experimental QTable if it exists
        if (rlService.getMultiDiscreteLearningProvider() != null) {
            Path qFile = getModelsDir().resolve(model.name + ".rqtb");
            try (ObjectOutputStream oosIdx = new ObjectOutputStream(
                    new BufferedOutputStream(Files.newOutputStream(qFile)))) {
                oosIdx.writeObject(rlService.getMultiDiscreteLearningProvider().getQTable());
            }
        }
    }

    public static RLModel loadModel(String name, RLTrainingService rlService) throws IOException, ClassNotFoundException {
        Path file = getModelsDir().resolve(name + ".rmeta");
        RLModel model;
        try (ObjectInputStream ois = new ObjectInputStream(
                new BufferedInputStream(Files.newInputStream(file)))) {
            model = (RLModel) ois.readObject();
        }
        Path netFile = getModelsDir().resolve(name + ".rnet");
        if (Files.exists(netFile)) {
            rlService.getAgent().load(netFile.toString());
        }
        
        // Load experimental QTable if it exists
        Path qFile = getModelsDir().resolve(name + ".rqtb");
        if (Files.exists(qFile) && rlService.getMultiDiscreteLearningProvider() != null) {
            try (ObjectInputStream oisIdx = new ObjectInputStream(
                    new BufferedInputStream(Files.newInputStream(qFile)))) {
                QTable qTable = (QTable) oisIdx.readObject();
                // Update baseline artifact only, do NOT switch active policy
                rlService.updateMultiDiscreteQTableBaseline(new QTableMultiDiscreteDecisionProvider(qTable));
            }
        }
        return model;
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
            boolean d3 = Files.deleteIfExists(getModelsDir().resolve(name + ".rqtb"));
            return d1 || d2 || d3;
        } catch (IOException e) {
            return false;
        }
    }

    public static RLModel createSnapshot(String name, RLTrainingService rlService,
                                        double logW, double costW, double raidW, double workingAreaW) {
        RLRewardConfig cfg = rlService.getRewardConfig();
        if (cfg == null) {
            cfg = RLRewardConfig.createDefault();
        }

        return new RLModel(name, rlService.getEpisodesTrained(),
                           rlService.getBestScore(), rlService.getEpsilon(),
                           logW, costW, raidW, workingAreaW,
                           cfg.clone());
    }

    public static void restoreFromModel(RLTrainingService rlService, RLModel model) {
        rlService.resetRuntimeState();
        rlService.setEpisodesTrained(model.episodesTrained);
        rlService.setEpsilon(model.epsilon);
        rlService.setBestScore(model.bestScore);
        if (model.rewardConfig != null) {
            rlService.setRewardConfig(model.rewardConfig.clone());
        }
    }
}
