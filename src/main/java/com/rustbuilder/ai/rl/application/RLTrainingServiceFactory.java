package com.rustbuilder.ai.rl.application;

@FunctionalInterface
public interface RLTrainingServiceFactory {
    RLTrainingService create();
}
