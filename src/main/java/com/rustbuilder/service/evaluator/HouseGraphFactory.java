package com.rustbuilder.service.evaluator;

import com.rustbuilder.service.graph.HouseGraph;

@FunctionalInterface
public interface HouseGraphFactory {
    HouseGraph create();
}
