package com.rustbuilder.model;

@FunctionalInterface
public interface GridModelFactory {

    GridModel create();

    static GridModelFactory defaultFactory() {
        return GridModel::new;
    }
}
