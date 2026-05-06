package com.rustbuilder.service;

import com.rustbuilder.ai.ea.GeneticAlgorithmService;

public class TestAI {
    public static void main(String[] args) {
        GeneticAlgorithmService ga = new GeneticAlgorithmService();
        ga.setPopulationSize(50);
        
        for (int i = 0; i < 500; i++) {
            ga.evolve(1, 0.22, 0.18, 0.28, 0.22, 0.10, null);
        }
        
        com.rustbuilder.ai.ea.BaseGenome best = ga.getBestGenome();
        
        com.rustbuilder.model.GridModel tempGrid = new com.rustbuilder.model.GridModel();
        best.decode(tempGrid);
    }
}
