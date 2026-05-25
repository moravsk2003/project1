package com.rustbuilder.ai.ea.application;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.rustbuilder.ai.ea.domain.BaseGenome;

public class GeneticAlgorithmServiceTest {

    private GeneticAlgorithmService gaService;

    @BeforeEach
    void setUp() {
        gaService = new GeneticAlgorithmService();
        gaService.setPopulationSize(20);
        gaService.setStagnationLimit(3); // Small stagnation limit for quick testing
        gaService.setMaxMutationRate(0.60);
        gaService.setMutationRate(0.15);
    }

    @Test
    void testInitializePopulationGuaranteesTCGenomes() {
        gaService.initializePopulation();
        List<BaseGenome> population = gaService.getPopulation();
        assertEquals(20, population.size());

        // At least 4 genomes should have TC since populationSize / 5 = 20 / 5 = 4
        int tcCount = 0;
        for (BaseGenome genome : population) {
            boolean hasTC = genome.getActions().stream()
                    .anyMatch(action -> action.actionType == com.rustbuilder.core.action.BuildAction.ActionType.TC);
            if (hasTC) {
                tcCount++;
            }
        }
        assertTrue(tcCount >= 4, "Expected at least 4 genomes with TC, got: " + tcCount);
    }

    @Test
    void testEvolveTriggersAdaptiveMutationAndIslandRestart() {
        // We will run evolution under zero weights to evaluate stagnation behaviour.
        // Since fitness won't improve (we will stub/run with zero weight or constant evaluations),
        // we can observe the stagnation counter increments, mutation rate boosts, and island restarts.
        gaService.initializePopulation();

        // 1. Initially, stagnation = 0, mutation rate = base rate (0.15)
        assertEquals(0, gaService.getStagnationCounter());
        assertEquals(0.15, gaService.getCurrentMutationRate(), 0.001);

        // Run evolution for 2 generations
        gaService.evolve(2, 0, 0, 0, 0, 0, null);

        // Since all evaluations have 0 fitness (or close to it), fitness won't improve beyond the first generation.
        // Let's verify stagnation increases.
        int currentStagnation = gaService.getStagnationCounter();
        assertTrue(currentStagnation > 0, "Stagnation counter should increase");

        // Run for more generations to trigger stagnation limit (3) and island restart (6)
        // With stagnationLimit = 3, after 3 generations of no improvement:
        // currentMutationRate should start boosting.
        // At stagnationCounter >= 6 (stagnationLimit * 2), island restart triggers.
        // Island restart resets stagnationCounter to 0 and currentMutationRate to base (0.15).
        
        boolean hitStagnationBoost = false;
        boolean hitIslandRestart = false;

        for (int i = 0; i < 15; i++) {
            gaService.evolve(1, 0, 0, 0, 0, 0, null);
            double mutationRate = gaService.getCurrentMutationRate();
            int stagnation = gaService.getStagnationCounter();

            if (mutationRate > 0.15 && stagnation >= 3) {
                hitStagnationBoost = true;
            }

            // Since it's evolved line-by-line, let's check if the stagnation counter resets
            // after hitting stagnationLimit * 2 = 6.
            // On the generation where islandRestart is true, it resets stagnationCounter to 0.
            if (stagnation == 0 && mutationRate == 0.15) {
                hitIslandRestart = true;
            }
        }

        assertTrue(hitStagnationBoost, "Adaptive mutation should boost mutation rate during stagnation");
        assertTrue(hitIslandRestart, "Island restart should trigger and reset stagnation/mutation rate");
    }
}
