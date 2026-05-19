package com.rustbuilder.ai.rl.ports;

import com.rustbuilder.ai.rl.environment.spec.EncodingRuntimeConfig;
import com.rustbuilder.ai.rl.environment.state.StateRepresentationEncoder;
import java.util.Objects;

public final class StateEncoderBundle {
    private final EncodingRuntimeConfig config;
    private final StateRepresentationEncoder encoder;

    public StateEncoderBundle(EncodingRuntimeConfig config, StateRepresentationEncoder encoder) {
        this.config = Objects.requireNonNull(config, "config");
        this.encoder = Objects.requireNonNull(encoder, "encoder");
    }

    public EncodingRuntimeConfig getConfig() {
        return config;
    }

    public StateRepresentationEncoder getEncoder() {
        return encoder;
    }
}
