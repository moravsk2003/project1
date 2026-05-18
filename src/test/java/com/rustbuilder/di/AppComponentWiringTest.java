package com.rustbuilder.di;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;

import org.junit.jupiter.api.Test;

class AppComponentWiringTest {

    @Test
    void createsIndependentRuntimeServicesThroughGuice() {
        AppComponent component = new AppComponent();

        assertNotNull(component.createGridModel());
        assertNotNull(component.createHouseEvaluator());
        assertNotNull(component.createGeneticAlgorithmService());
        assertNotSame(component.createHouseEvaluator(), component.createHouseEvaluator());
    }
}
