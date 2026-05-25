package com.rustbuilder.model;

import static org.junit.jupiter.api.Assertions.*;

import com.rustbuilder.model.core.*;
import com.rustbuilder.model.structure.*;
import com.rustbuilder.config.GameConstants;
import java.util.List;
import java.util.concurrent.*;
import org.junit.jupiter.api.Test;

class BoundaryAndStressTest {

    @Test
    void extremeHeightsTest() {
        GridModel grid = new GridModel();

        // Normal base at Z=0 to provide support
        Foundation f1 = new Foundation(0, 0, 0);
        assertTrue(grid.addBlock(f1));

        // Add blocks at high floors
        // Note: Z is packed as (z + 128L) & 0xFF. Range of Z without wrapping is -128 to 127.
        // Let's place a block at Z = 100 and Z = -50 (within range) and Z = 500 (causes wrap-around).
        Foundation fHigh = new Foundation(0, 0, 100);
        assertTrue(grid.addBlockSilent(fHigh));

        Foundation fLow = new Foundation(0, 0, -50);
        assertTrue(grid.addBlockSilent(fLow));

        Foundation fWrap = new Foundation(0, 0, 500); // 500 wraps to (500+128)&0xFF
        assertTrue(grid.addBlockSilent(fWrap));

        // Verify retrieval via getNearbyBlocks
        List<BuildingBlock> nearbyHigh = grid.getNearbyBlocks(0, 0, 100, 1.0);
        assertTrue(nearbyHigh.contains(fHigh));

        List<BuildingBlock> nearbyLow = grid.getNearbyBlocks(0, 0, -50, 1.0);
        assertTrue(nearbyLow.contains(fLow));

        List<BuildingBlock> nearbyWrap = grid.getNearbyBlocks(0, 0, 500, 1.0);
        assertTrue(nearbyWrap.contains(fWrap));
    }

    @Test
    void boundaryCoordinatesTest() {
        GridModel grid = new GridModel();

        // gx = coord / TILE_SIZE. Offset is 500000. Under 0xFFFFF (1048576) mask, 
        // max gx/gy supported without wrapping is 548576. 
        // With TILE_SIZE = 60, max coordinate is approx 32,914,560.
        // Let's test boundary coordinates:
        double xMax = 30_000_000.0;
        double yMax = 30_000_000.0;
        double xMin = -29_000_000.0;
        double yMin = -29_000_000.0;

        Foundation fMax = new Foundation(xMax, yMax, 0);
        Foundation fMin = new Foundation(xMin, yMin, 0);

        assertTrue(grid.addBlockSilent(fMax));
        assertTrue(grid.addBlockSilent(fMin));

        List<BuildingBlock> nearbyMax = grid.getNearbyBlocks(xMax, yMax, 0, 1.0);
        assertTrue(nearbyMax.contains(fMax));

        List<BuildingBlock> nearbyMin = grid.getNearbyBlocks(xMin, yMin, 0, 1.0);
        assertTrue(nearbyMin.contains(fMin));
    }

    @Test
    void hugeStructureStressTest() {
        GridModel grid = new GridModel();

        // 1. Build a massive foundation grid (e.g. 20 x 20) at Z=0
        // TILE_SIZE is 60.
        int size = 20;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                Foundation f = new Foundation(i * GameConstants.TILE_SIZE, j * GameConstants.TILE_SIZE, 0);
                grid.addBlockSilent(f);
            }
        }

        // 2. Build walls on top of it up to 5 floors high (starting from z=0)
        for (int z = 0; z <= 4; z++) {
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    // Place wall at coordinate
                    Wall w = new Wall(i * GameConstants.TILE_SIZE, j * GameConstants.TILE_SIZE, z, Orientation.NORTH);
                    grid.addBlockSilent(w);
                }
            }
        }

        // 3. Verify total number of blocks
        // 20*20 foundations + 20*20*5 walls = 400 + 2000 = 2400 blocks.
        assertEquals(2400, grid.getAllBlocks().size());

        // 4. Finalize load - runs stability updates and removes unsupported blocks.
        // Since we built everything sitting on Z=0, stability should keep everything stable.
        long startTime = System.nanoTime();
        grid.finalizeLoad();
        long endTime = System.nanoTime();
        double durationMs = (endTime - startTime) / 1_000_000.0;
        
        System.out.printf("Stress test: 2400 blocks finalized in %.2f ms%n", durationMs);

        // Verify that everything is still present (stable)
        assertEquals(2400, grid.getAllBlocks().size());

        // 5. Query nearby blocks in a hot spot
        startTime = System.nanoTime();
        for (int k = 0; k < 1000; k++) {
            grid.getNearbyBlocks(10 * GameConstants.TILE_SIZE, 10 * GameConstants.TILE_SIZE, 3, 1.0);
        }
        endTime = System.nanoTime();
        double queryDurationMs = (endTime - startTime) / 1_000_000.0;
        System.out.printf("Stress test: 1000 nearby queries executed in %.2f ms%n", queryDurationMs);
    }

    @Test
    void concurrentOperationsTest() throws InterruptedException, ExecutionException {
        // Since GridModel instances are independent and do not share mutable static state,
        // we should be able to manipulate separate GridModel instances concurrently on different threads.
        int threadsCount = 8;
        ExecutorService executor = Executors.newFixedThreadPool(threadsCount);
        
        List<Callable<Integer>> tasks = new java.util.ArrayList<>();
        for (int t = 0; t < threadsCount; t++) {
            final int threadId = t;
            tasks.add(() -> {
                GridModel localGrid = new GridModel();
                // Create a small base
                for (int i = 0; i < 5; i++) {
                    for (int j = 0; j < 5; j++) {
                        Foundation f = new Foundation(i * GameConstants.TILE_SIZE, j * GameConstants.TILE_SIZE, 0);
                        localGrid.addBlockSilent(f);
                    }
                }
                localGrid.finalizeLoad();
                return localGrid.getAllBlocks().size();
            });
        }

        List<Future<Integer>> futures = executor.invokeAll(tasks);
        for (Future<Integer> future : futures) {
            assertEquals(25, future.get());
        }
        
        executor.shutdown();
        assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
    }
}
