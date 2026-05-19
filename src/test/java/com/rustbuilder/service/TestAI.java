package com.rustbuilder.service;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import com.rustbuilder.ai.ea.domain.BaseGenome;
import com.rustbuilder.ai.ea.application.GeneticAlgorithmService;
import com.rustbuilder.model.GridModel;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

@Disabled("Manual GA smoke test: runs many evolution iterations and is too slow for regular test runs.")
class TestAI {

    @Test
    void evolvesAndDecodesBestGenome() {
        GeneticAlgorithmService ga = new GeneticAlgorithmService();
        ga.setPopulationSize(50);

        assertDoesNotThrow(() -> {
            for (int i = 0; i < 500; i++) {
                ga.evolve(1, 0.22, 0.18, 0.28, 0.22, 0.10, null);
            }
        });

        BaseGenome best = ga.getBestGenome();
        assertNotNull(best);

        GridModel tempGrid = new GridModel();
        assertDoesNotThrow(() -> best.decode(tempGrid));
    }
}
