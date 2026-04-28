package com.rustbuilder.ai.ea;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.service.evaluator.HouseEvaluator;

/**
 * Genetic Algorithm engine for evolving house designs.
 * Uses tournament selection, single-point crossover, and mutation.
 *
 * <p>Supports adaptive mutation: when the best fitness does not improve
 * for {@code stagnationLimit} generations, the effective mutation rate is
 * temporarily boosted, then gradually restored to the base rate.</p>
 */
public class GeneticAlgorithmService {

    private static final Random RNG = new Random();

    // ── Configurable hyperparameters (with recommended defaults) ──────────────
    private int    populationSize   = 80;   // recommended: 50–200
    private int    tournamentSize   = 4;    // recommended: 3–8
    private int    eliteCount       = 5;    // recommended: 3–10
    private int    freshBloodCount  = 3;    // random genomes injected per gen
    private double baseMutationRate = 0.15; // initial / target mutation rate

    // ── Adaptive mutation ─────────────────────────────────────────────────────
    /** Generations without improvement before boosting mutation. */
    private int    stagnationLimit  = 15;   // recommended: 10–30
    /** Maximum mutation rate reached during a stagnation burst. */
    private double maxMutationRate  = 0.60; // recommended: 0.40–0.80
    /** How fast the rate recovers toward baseMutationRate after a burst (0–1). */
    private final double adaptiveCooldown = 0.90; // 0.90 = 10% closer to base each gen

    // ── Runtime state ─────────────────────────────────────────────────────────
    private double currentMutationRate;
    private int    stagnationCounter = 0;

    private List<BaseGenome> population;
    private int    generation;
    private double bestFitness;
    private BaseGenome bestGenome;

    private final HouseEvaluator evaluator;

    /** Fitness cache: avoids re-evaluating identical genomes within the same weight config. */
    private final Map<BaseGenome, Double> fitnessCache = new ConcurrentHashMap<>();
    private int lastWeightsHash = 0;

    /**
     * Cached evaluation result of the current best genome.
     * Set by the training thread; read by the UI thread via updateStatsLabels.
     * Volatile ensures visibility across threads without locking.
     */
    private volatile HouseEvaluator.EvaluationResult lastBestResult = null;

    public GeneticAlgorithmService() {
        this.evaluator = new HouseEvaluator();
        this.population = new ArrayList<>();
        this.generation = 0;
        this.bestFitness = -1;
        this.currentMutationRate = baseMutationRate;
    }

    // ── Population init ───────────────────────────────────────────────────────

    public void initializePopulation() {
        population.clear();
        fitnessCache.clear();
        generation = 0;
        bestFitness = -1;
        bestGenome = null;
        stagnationCounter = 0;
        currentMutationRate = baseMutationRate;
        for (int i = 0; i < populationSize; i++) {
            BaseGenome g = BaseGenome.randomGenome();
            // Fix: guarantee ~20 % of the initial population has a TC action,
            // otherwise logistics score = 0 for most genomes and the fitness
            // landscape is flat — the GA cannot differentiate good from bad.
            if (i < populationSize / 5) {
                g = BaseGenome.randomGenomeWithTC();
            }
            population.add(g);
        }
    }

    // ── Evaluation ────────────────────────────────────────────────────────────

    private void evaluatePopulation(double logW, double costW, double raidW, double workingAreaW) {
        int currentWeightsHash = Objects.hash(logW, costW, raidW, workingAreaW);
        if (currentWeightsHash != lastWeightsHash) {
            fitnessCache.clear();
            // BUG FIX: Force re-evaluation of all surviving/elite genomes with the new weights
            for (BaseGenome g : population) {
                g.setFitness(-1);
            }
            lastWeightsHash = currentWeightsHash;
        }

        evaluator.setWeights(logW, costW, raidW, workingAreaW);

        population.parallelStream().forEach(genome -> {
            if (genome.getFitness() >= 0) return;

            if (fitnessCache.containsKey(genome)) {
                genome.setFitness(fitnessCache.get(genome));
                return;
            }

            GridModel tempGrid = new GridModel();
            genome.decode(tempGrid);
            int blockCount = tempGrid.getAllBlocks().size();

            // Too few blocks to evaluate properly
            if (blockCount < 5) {
                genome.setFitness(0.0);
                fitnessCache.put(genome, 0.0);
                return;
            }

            try {
                HouseEvaluator.EvaluationResult result = evaluator.evaluate(tempGrid);

                // ── Key-component bonuses ─────────────────────────────────────
                double bonus = 0;
                boolean hasTC   = tempGrid.getAllBlocks().stream()
                    .anyMatch(b -> b.getType() == com.rustbuilder.model.core.BuildingType.TC);
                boolean hasLoot = tempGrid.getAllBlocks().stream()
                    .anyMatch(b -> b.getType() == com.rustbuilder.model.core.BuildingType.LOOT_ROOM);
                boolean hasWB   = tempGrid.getAllBlocks().stream()
                    .anyMatch(b -> b.getType() == com.rustbuilder.model.core.BuildingType.WORKBENCH);

                if (hasTC)   bonus += 0.10;
                if (hasLoot) bonus += 0.05;
                if (hasWB)   bonus += 0.05;

                // ── Combined fitness ──────────────────────────────────────────
                // evalFitness  : weighted score from logistics / cost / raid
                // bonus        : reward for key strategic components
                double finalFitness = result.finalScore + bonus;

                genome.setFitness(finalFitness);
                fitnessCache.put(genome, finalFitness);
            } catch (Exception e) {
                // Evaluation crashed — keep genome at 0 fitness
                genome.setFitness(0.0);
                fitnessCache.put(genome, 0.0);
            }
        });
    }

    // ── Selection ─────────────────────────────────────────────────────────────

    private BaseGenome tournamentSelect() {
        BaseGenome best = null;
        for (int i = 0; i < tournamentSize; i++) {
            BaseGenome candidate = population.get(RNG.nextInt(population.size()));
            if (best == null || candidate.getFitness() > best.getFitness()) {
                best = candidate;
            }
        }
        return best;
    }

    // ── Adaptive mutation ─────────────────────────────────────────────────────

    /**
     * Updates {@code currentMutationRate} based on stagnation.
     * If fitness improved this generation → reset counter and cool down rate.
     * If stagnation threshold reached → boost rate toward {@code maxMutationRate}.
     */
    private void updateAdaptiveMutation(boolean improved) {
        if (improved) {
            stagnationCounter = 0;
            // Cool down toward base rate
            currentMutationRate = currentMutationRate * adaptiveCooldown
                                + baseMutationRate * (1 - adaptiveCooldown);
        } else {
            stagnationCounter++;
            if (stagnationCounter >= stagnationLimit) {
                // Linear interpolation: the longer the stagnation, the higher the boost
                double excess = Math.min(stagnationCounter - stagnationLimit, stagnationLimit);
                double t = excess / stagnationLimit; // 0..1
                currentMutationRate = baseMutationRate + t * (maxMutationRate - baseMutationRate);
            }
        }
        // Hard clamp
        currentMutationRate = Math.max(baseMutationRate, Math.min(maxMutationRate, currentMutationRate));
    }

    // ── Evolution ─────────────────────────────────────────────────────────────

    /**
     * Evolve the population for the given number of generations.
     *
     * @param generations      number of generations to run
     * @param logW             logistics weight
     * @param costW            cost weight
     * @param raidW            raid resistance weight
     * @param workingAreaW     working-area weight
     * @param progressCallback called after each generation with (currentGen, bestFitness, currentMutationRate)
     */
    public void evolve(int generations, double logW, double costW, double raidW, double workingAreaW,
                       Consumer<double[]> progressCallback) {
        if (population.isEmpty()) {
            initializePopulation();
        }

        for (int g = 0; g < generations; g++) {
            evaluatePopulation(logW, costW, raidW, workingAreaW);

            population.sort(Comparator.comparingDouble(BaseGenome::getFitness).reversed());

            boolean improved = population.get(0).getFitness() > bestFitness;
            if (improved) {
                bestFitness = population.get(0).getFitness();
                bestGenome  = population.get(0).copy();
                // Cache the evaluation detail of the new best genome so the UI
                // can display raid/cost/logistics stats without re-decoding.
                try {
                    GridModel tempGrid = new GridModel();
                    bestGenome.decode(tempGrid);
                    lastBestResult = evaluator.evaluate(tempGrid);
                } catch (Exception ignored) {
                    lastBestResult = null;
                }
            }

            // Update adaptive mutation BEFORE building next generation
            updateAdaptiveMutation(improved);

            List<BaseGenome> nextGen = new ArrayList<>();

            // Elitism
            for (int i = 0; i < eliteCount && i < population.size(); i++) {
                nextGen.add(population.get(i).copy());
            }

            // Fresh blood — more when stagnating
            int inject = freshBloodCount + (stagnationCounter >= stagnationLimit ? 2 : 0);
            for (int i = 0; i < inject; i++) {
                nextGen.add(BaseGenome.randomGenome());
            }

            // Island restart: if stagnation persists past 2× the limit,
            // replace the bottom half of the population with fresh genomes.
            // This prevents total convergence while preserving the elite top half.
            boolean islandRestart = stagnationCounter >= stagnationLimit * 2;
            int islandSlots = islandRestart ? (populationSize / 2 - nextGen.size()) : 0;
            for (int i = 0; i < islandSlots; i++) {
                nextGen.add(BaseGenome.randomGenomeWithTC());
            }

            // CRITICAL FIX: The Death Spiral
            // If we hit island restart, we MUST reset the stagnation counter and mutation rate.
            // Otherwise, mutation stays at 100% forever, every child is completely randomized, 
            // and they will never beat the elite, causing infinite stagnation.
            if (islandRestart) {
                stagnationCounter = 0;
                currentMutationRate = baseMutationRate;
            }

            // Crossover + mutation
            while (nextGen.size() < populationSize) {
                BaseGenome parent1 = tournamentSelect();
                BaseGenome parent2 = tournamentSelect();
                BaseGenome child   = BaseGenome.crossover(parent1, parent2);
                child.mutate(currentMutationRate);
                child.setFitness(-1);
                nextGen.add(child);
            }

            population = nextGen;
            generation++;

            if (progressCallback != null) {
                progressCallback.accept(new double[]{generation, bestFitness, currentMutationRate, stagnationCounter});
            }
        }

        // Final evaluation
        evaluatePopulation(logW, costW, raidW, workingAreaW);
        population.sort(Comparator.comparingDouble(BaseGenome::getFitness).reversed());
        if (population.get(0).getFitness() > bestFitness) {
            bestFitness = population.get(0).getFitness();
            bestGenome  = population.get(0).copy();
        }
    }

    // ── Getters / Setters ─────────────────────────────────────────────────────

    public BaseGenome getBestGenome()  { return bestGenome; }
    public double     getBestFitness() { return bestFitness; }
    public int        getGeneration()  { return generation; }
    public double     getCurrentMutationRate() { return currentMutationRate; }
    public int        getStagnationCounter()   { return stagnationCounter; }
    public HouseEvaluator.EvaluationResult getLastBestResult() { return lastBestResult; }

    public List<BaseGenome> getPopulation()              { return population; }
    public void setPopulation(List<BaseGenome> population) { this.population = population; }
    public void setGeneration(int generation)             { this.generation = generation; }
    public void setBestFitness(double bestFitness)        { this.bestFitness = bestFitness; }
    public void setBestGenome(BaseGenome bestGenome)      { this.bestGenome = bestGenome; }

    /** Sets the base mutation rate and resets the adaptive state. */
    public void setMutationRate(double rate) {
        this.baseMutationRate   = rate;
        this.currentMutationRate = rate;
        this.stagnationCounter  = 0;
    }

    public void setPopulationSize(int size)    { this.populationSize  = size; }
    public void setTournamentSize(int size)    { this.tournamentSize  = size; }
    public void setEliteCount(int count)       { this.eliteCount      = count; }
    public void setFreshBloodCount(int count)  { this.freshBloodCount = count; }
    public void setStagnationLimit(int limit)  { this.stagnationLimit = limit; }
    public void setMaxMutationRate(double rate){ this.maxMutationRate  = rate; }

    public int    getPopulationSize()  { return populationSize; }
    public int    getTournamentSize()  { return tournamentSize; }
    public int    getEliteCount()      { return eliteCount; }
    public int    getFreshBloodCount() { return freshBloodCount; }
    public int    getStagnationLimit() { return stagnationLimit; }
    public double getMaxMutationRate() { return maxMutationRate; }
    public double getBaseMutationRate(){ return baseMutationRate; }
}
