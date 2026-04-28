package com.rustbuilder.model.core;

/**
 * Rust door types with their sulfur raid costs (cheapest method).
 */
public enum DoorType {
    SHEET_METAL(1575, "Sheet Metal Door"),
    GARAGE(3200, "Garage Door"),
    ARMORED(5150, "Armored Door");

    private final int sulfurCost;
    private final String displayName;

    DoorType(int sulfurCost, String displayName) {
        this.sulfurCost = sulfurCost;
        this.displayName = displayName;
    }

    public int getSulfurCost() {
        return sulfurCost;
    }

    public String getDisplayName() {
        return displayName;
    }
}
