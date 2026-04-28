package com.rustbuilder.service.generator;

import com.rustbuilder.model.structure.Foundation;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.Orientation;
import com.rustbuilder.model.structure.Wall;

public class GeneratorService {

    private final GridModel gridModel;
    private static final double TILE_SIZE = com.rustbuilder.config.GameConstants.TILE_SIZE;

    public GeneratorService(GridModel gridModel) {
        this.gridModel = gridModel;
    }

    public void generateSimpleBase(double startX, double startY) {
        gridModel.clear();

        // 1. Core: 2x2 Square
        for (int x = 0; x < 2; x++) {
            for (int y = 0; y < 2; y++) {
                gridModel.addBlock(new Foundation(startX + x * TILE_SIZE, startY + y * TILE_SIZE, 0));
            }
        }

        // 2. Core Walls (Inner shell)
        // Outer walls of the 2x2 core
        // Top-Left (0,0) West Wall
        gridModel.addBlock(new Wall(startX, startY, 0, Orientation.WEST));
        // Top-Left (0,0) North Wall
        gridModel.addBlock(new Wall(startX, startY, 0, Orientation.NORTH));

        // Top-Right (1,0) East Wall
        gridModel.addBlock(new Wall(startX + TILE_SIZE, startY, 0, Orientation.EAST));
        // Top-Right (1,0) North Wall
        gridModel.addBlock(new Wall(startX + TILE_SIZE, startY, 0, Orientation.NORTH));

        // Bottom-Left (0,1) West Wall
        gridModel.addBlock(new Wall(startX, startY + TILE_SIZE, 0, Orientation.WEST));
        // Bottom-Left (0,1) South Wall
        gridModel.addBlock(new Wall(startX, startY + TILE_SIZE, 0, Orientation.SOUTH));

        // Bottom-Right (1,1) East Wall
        gridModel.addBlock(new Wall(startX + TILE_SIZE, startY + TILE_SIZE, 0, Orientation.EAST));
        // Bottom-Right (1,1) South Wall
        gridModel.addBlock(new Wall(startX + TILE_SIZE, startY + TILE_SIZE, 0, Orientation.SOUTH));

        // 3. Honeycomb (Triangle foundations around the core)
        // Simplified for now: Adding a layer of square foundations

        for (int x = -1; x < 3; x++) {
            for (int y = -1; y < 3; y++) {
                // Skip the inner 2x2
                if (x >= 0 && x < 2 && y >= 0 && y < 2)
                    continue;

                gridModel.addBlock(new Foundation(startX + x * TILE_SIZE, startY + y * TILE_SIZE, 0));
            }
        }

        // Add outer walls for the honeycomb
        // Top edge
        for (int x = -1; x < 3; x++)
            gridModel.addBlock(new Wall(startX + x * TILE_SIZE, startY - TILE_SIZE, 0, Orientation.NORTH));
        // Bottom edge
        for (int x = -1; x < 3; x++)
            gridModel.addBlock(new Wall(startX + x * TILE_SIZE, startY + 2 * TILE_SIZE, 0, Orientation.SOUTH));
        // Left edge
        for (int y = -1; y < 3; y++)
            gridModel.addBlock(new Wall(startX - TILE_SIZE, startY + y * TILE_SIZE, 0, Orientation.WEST));
        // Right edge
        for (int y = -1; y < 3; y++)
            gridModel.addBlock(new Wall(startX + 2 * TILE_SIZE, startY + y * TILE_SIZE, 0, Orientation.EAST));
    }
}
