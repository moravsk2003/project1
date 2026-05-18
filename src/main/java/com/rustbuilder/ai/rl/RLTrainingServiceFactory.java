package com.rustbuilder.ai.rl;

@FunctionalInterface
public interface RLTrainingServiceFactory {
    RLTrainingService create();
}
