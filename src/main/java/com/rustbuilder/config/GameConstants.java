package com.rustbuilder.config;

public class GameConstants {
    public static final double TILE_SIZE = 60.0;
    public static final double HALF_TILE = TILE_SIZE / 2.0;
    public static final double WALL_THICKNESS = 6.0;
    public static final double WALL_OFFSET = WALL_THICKNESS / 2.0;
    public static final double SNAP_RADIUS = 45.0;
    public static final double TRIANGLE_HEIGHT = TILE_SIZE * Math.sqrt(3) / 2.0;
    public static final double TRIANGLE_OFFSET = TILE_SIZE - TRIANGLE_HEIGHT;

    /** X origin of the AI/EA grid coordinate system (world-space pixels). */
    public static final double GRID_ORIGIN_X = 200.0;
    /** Y origin of the AI/EA grid coordinate system (world-space pixels). */
    public static final double GRID_ORIGIN_Y = 200.0;
}
