package com.rustbuilder.ai.core;

import com.rustbuilder.ai.ea.domain.BaseGenome;
import com.rustbuilder.ai.ea.application.GeneticAlgorithmService;

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

/**
 * Manages AI model persistence — save/load trained populations to disk.
 * Models are stored as serialized .dat files in the models/ directory.
 */
public class AIModelManager {

    private static final String MODELS_DIR = "models";

    /**
     * Saved model data — serializable snapshot of training state.
     */
    public static class AIModel implements Serializable {
        private static final long serialVersionUID = 1L;

        public final String name;
        public final List<BaseGenome> population;
        public final int generation;
        public final double bestFitness;
        public final BaseGenome bestGenome;
        public final double logisticsWeight;
        public final double costWeight;
        public final double raidWeight;
        public final double workingAreaWeight;
        public final double safeZoneWeight;

        public AIModel(String name, List<BaseGenome> population, int generation,
                       double bestFitness, BaseGenome bestGenome,
                       double logisticsWeight, double costWeight, double raidWeight, double workingAreaWeight,
                       double safeZoneWeight) {
            this.name = name;
            this.population = population;
            this.generation = generation;
            this.bestFitness = bestFitness;
            this.bestGenome = bestGenome;
            this.logisticsWeight = logisticsWeight;
            this.costWeight = costWeight;
            this.raidWeight = raidWeight;
            this.workingAreaWeight = workingAreaWeight;
            this.safeZoneWeight = safeZoneWeight;
        }

        @Override
        public String toString() {
            return String.format("%s (Gen %d, Best: %.3f)", name, generation, bestFitness);
        }
    }

    /**
     * Ensure models directory exists.
     */
    private static Path getModelsDir() {
        Path dir = Paths.get(MODELS_DIR);
        try {
            Files.createDirectories(dir);
        } catch (IOException e) {
        }
        return dir;
    }

    /**
     * Save an AI model to disk.
     */
    public static void saveModel(AIModel model) throws IOException {
        Path file = getModelsDir().resolve(model.name + ".dat");
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new BufferedOutputStream(Files.newOutputStream(file)))) {
            oos.writeObject(model);
        }
    }

    /**
     * Load an AI model from disk.
     */
    public static AIModel loadModel(String name) throws IOException, ClassNotFoundException {
        Path file = getModelsDir().resolve(name + ".dat");
        try (ObjectInputStream ois = new ObjectInputStream(
                new BufferedInputStream(Files.newInputStream(file)))) {
            return (AIModel) ois.readObject();
        }
    }

    /**
     * List all available model names.
     */
    public static List<String> listModels() {
        Path dir = getModelsDir();
        try (Stream<Path> files = Files.list(dir)) {
            return files
                .filter(p -> p.toString().endsWith(".dat"))
                .map(p -> {
                    String fileName = p.getFileName().toString();
                    return fileName.substring(0, fileName.length() - 4);
                })
                .sorted()
                .collect(Collectors.toList());
        } catch (IOException e) {
            return new ArrayList<>();
        }
    }

    /**
     * Delete a saved model.
     */
    public static boolean deleteModel(String name) {
        Path file = getModelsDir().resolve(name + ".dat");
        try {
            return Files.deleteIfExists(file);
        } catch (IOException e) {
            return false;
        }
    }

    /**
     * Create an AIModel snapshot from the current GA state.
     */
    public static AIModel createSnapshot(String name, GeneticAlgorithmService ga,
                                          double logW, double costW, double raidW, double workingAreaW,
                                          double safeZoneW) {
        return new AIModel(name, ga.getPopulation(), ga.getGeneration(),
                           ga.getBestFitness(), ga.getBestGenome(),
                           logW, costW, raidW, workingAreaW, safeZoneW);
    }

    /**
     * Restore GA state from a loaded model.
     */
    public static void restoreFromModel(GeneticAlgorithmService ga, AIModel model) {
        ga.setPopulation(model.population);
        ga.setGeneration(model.generation);
        ga.setBestFitness(model.bestFitness);
        ga.setBestGenome(model.bestGenome);
    }
}
