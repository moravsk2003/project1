package com.rustbuilder.service.physics;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.Socket;
import com.rustbuilder.model.structure.Foundation;

public class SocketPlacementResolverTest {

    private GridModel gridModel;

    @BeforeEach
    void setUp() {
        gridModel = new GridModel();
    }

    @Test
    void testResolvePrefersHigherPriorityWhenDistancesAreEqual() {
        // Place foundation at 0,0.
        // Sockets: North (30,0, side 0), East (60,30, side 1), South (30,60, side 2), West (0,30, side 3)
        Foundation foundation = new Foundation(0, 0, 0, 0);
        gridModel.addBlock(foundation);

        // Aim point (60, 0) is equidistant to North socket (30, 0) and East socket (60, 30) (dx=30, dy=30).
        double aimX = 60;
        double aimY = 0;

        // Custom preference: East (side 1) has priority 10, North (side 0) has priority 2.
        SocketPlacementResolver.SocketPreference preferEast = (block, socket) -> {
            if (socket.getSide() == 1) return 10;
            if (socket.getSide() == 0) return 2;
            return 0;
        };

        SocketPlacementResolver.Result resultEast = SocketPlacementResolver.resolve(
                gridModel, aimX, aimY, "FOUNDATION", 0, false, preferEast);

        assertNotNull(resultEast.socket);
        assertEquals(1, resultEast.socket.getSide(), "Should select East socket (side 1) due to higher priority");

        // Custom preference: North (side 0) has priority 10, East (side 1) has priority 2.
        SocketPlacementResolver.SocketPreference preferNorth = (block, socket) -> {
            if (socket.getSide() == 0) return 10;
            if (socket.getSide() == 1) return 2;
            return 0;
        };

        SocketPlacementResolver.Result resultNorth = SocketPlacementResolver.resolve(
                gridModel, aimX, aimY, "FOUNDATION", 0, false, preferNorth);

        assertNotNull(resultNorth.socket);
        assertEquals(0, resultNorth.socket.getSide(), "Should select North socket (side 0) due to higher priority");
    }
}
