package com.rustbuilder.ai.rl;

public interface StateEncoderFactory {
    StateEncoderBundle create(RLTrainingService.EncoderMode mode);
}
