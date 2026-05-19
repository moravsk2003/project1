package com.rustbuilder.ai.rl.ports;


import com.rustbuilder.ai.rl.application.RLTrainingService;
public interface StateEncoderFactory {
    StateEncoderBundle create(RLTrainingService.EncoderMode mode);
}
