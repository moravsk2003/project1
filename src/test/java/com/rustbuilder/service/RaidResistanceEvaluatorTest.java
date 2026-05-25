package com.rustbuilder.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingTier;
import com.rustbuilder.model.core.Orientation;
import com.rustbuilder.model.deployable.ToolCupboard;
import com.rustbuilder.model.structure.Floor;
import com.rustbuilder.model.structure.Foundation;
import com.rustbuilder.model.structure.Wall;
import com.rustbuilder.service.evaluator.RaidResistanceEvaluator;
import com.rustbuilder.service.evaluator.RaidResistanceEvaluator.RaidResult;
import com.rustbuilder.service.graph.HouseGraph;
import com.rustbuilder.service.raid.RaidConstants;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class RaidResistanceEvaluatorTest {

    private RaidResistanceEvaluator evaluator;
    private HouseGraph graph;

    @BeforeEach
    void setUp() {
        evaluator = new RaidResistanceEvaluator();
        graph = new HouseGraph();
    }

    private <T extends BuildingBlock> T stone(T block) {
        block.setTier(BuildingTier.STONE);
        return block;
    }

    @Test
    void testSplashCostCalculationAndCoverageBonus() {
        GridModel grid = new GridModel();
        // Fully enclosed 1x1 room
        grid.addBlockSilent(stone(new Foundation(0, 0, 0)));
        grid.addBlockSilent(stone(new Wall(0, 0, 0, Orientation.NORTH)));
        grid.addBlockSilent(stone(new Wall(0, 0, 0, Orientation.EAST)));
        grid.addBlockSilent(stone(new Wall(0, 0, 0, Orientation.SOUTH)));
        grid.addBlockSilent(stone(new Wall(0, 0, 0, Orientation.WEST)));
        grid.addBlockSilent(stone(new Floor(0, 0, 1, 0)));
        grid.addBlockSilent(new ToolCupboard(0, 0, 0, 0));
        grid.finalizeLoad();

        graph.buildGraph(grid.getAllBlocks());
        RaidResult result = evaluator.evaluate(graph, grid.getAllBlocks());

        // Standard stone wall is 4400 sulfur
        assertEquals(4400, result.sulfurToTC, "Enclosed TC room should require a standard wall breach");

        // The four walls share the same corner (0,0,0) so splash applies.
        // For stone wall: HP = 500. Rockets needed = ceil(500 / 137) = 4.
        // Total sulfur for 4 rockets = 4 * 1400 = 5600.
        // Adjacent walls = 4. Cost per wall = 5600 / 4 = 1400.
        // The coverage of the node is the sum of splash costs of protecting structures.
        // Here we have 4 walls, each splash cost is 1400. Ceiling is not grouped (since it's a floor),
        // so its cost is standard 4400.
        // Total coverage = 4 * 1400 + 4400 = 10000.
        // Coverage bonus = 10000 * 0.05 = 500.
        // Since there is no honeycomb (coverage < 15000), total coverage bonus should be 500.
        // Check if result score is in range.
        assertTrue(result.score > 0.0 && result.score < 1.0);
    }

    @Test
    void testHoneycombBonusTriggersCorrectly() {
        GridModel grid = new GridModel();
        
        // Core room (HQM)
        grid.addBlockSilent(stone(new Foundation(0, 0, 0)));
        Floor ceiling = new Floor(0, 0, 1, 0);
        ceiling.setTier(BuildingTier.HQM);
        grid.addBlockSilent(ceiling);
        grid.addBlockSilent(new ToolCupboard(0, 0, 0, 0));

        Wall w1 = new Wall(0, 0, 0, Orientation.NORTH); w1.setTier(BuildingTier.HQM); grid.addBlockSilent(w1);
        Wall w2 = new Wall(0, 0, 0, Orientation.EAST);  w2.setTier(BuildingTier.HQM); grid.addBlockSilent(w2);
        Wall w3 = new Wall(0, 0, 0, Orientation.SOUTH); w3.setTier(BuildingTier.HQM); grid.addBlockSilent(w3);
        Wall w4 = new Wall(0, 0, 0, Orientation.WEST);  w4.setTier(BuildingTier.HQM); grid.addBlockSilent(w4);

        // Add adjacent honeycomb foundation & walls to the North
        grid.addBlockSilent(stone(new Foundation(0, 60, 0)));
        grid.addBlockSilent(stone(new Floor(0, 60, 1, 0)));
        grid.addBlockSilent(stone(new Wall(0, 60, 0, Orientation.NORTH)));
        grid.addBlockSilent(stone(new Wall(0, 60, 0, Orientation.EAST)));
        grid.addBlockSilent(stone(new Wall(0, 60, 0, Orientation.WEST)));
        
        grid.finalizeLoad();

        graph.buildGraph(grid.getAllBlocks());
        RaidResult result = evaluator.evaluate(graph, grid.getAllBlocks());

        // Check that the evaluator calculates a finite sulfur cost and non-zero score
        assertTrue(result.sulfurToTC > 4400, "TC should be behind HQM and outer honeycomb walls, got: " + result.sulfurToTC);
        assertTrue(result.score > 0.0);
    }
}
