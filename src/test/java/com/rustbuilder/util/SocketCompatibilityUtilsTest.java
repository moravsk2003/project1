package com.rustbuilder.util;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.Test;
import com.rustbuilder.model.core.Socket;

public class SocketCompatibilityUtilsTest {

    @Test
    void testIsEdgeSocket() {
        Socket center = new Socket(0, 0, 0, Socket.CENTER_SIDE);
        Socket edge = new Socket(0, 0, 0, 0);

        assertFalse(SocketCompatibilityUtils.isEdgeSocket(center));
        assertTrue(SocketCompatibilityUtils.isEdgeSocket(edge));
        assertFalse(SocketCompatibilityUtils.isEdgeSocket(null));
    }

    @Test
    void testAreEdgesParallel() {
        // Parallel sockets on opposite sides or aligned
        Socket socketA = new Socket(0, 0, 0, 0); // side 0: rotation = 0 -> tangent = 0
        Socket socketB = new Socket(0, 0, 180, 2); // side 2: rotation = 180 -> tangent = 180 -> normalized 0

        assertTrue(SocketCompatibilityUtils.areEdgesParallel(socketA, socketB));

        // Non-parallel sockets
        Socket socketC = new Socket(0, 0, 0, 1); // side 1: rotation = 0 -> tangent = 90
        assertFalse(SocketCompatibilityUtils.areEdgesParallel(socketA, socketC));
    }

    @Test
    void testAreEdgeSocketsConnected() {
        Socket socketA = new Socket(10.0, 10.0, 0, 0);
        Socket socketB = new Socket(10.5, 10.5, 180, 2);

        // Distance squared: 0.5^2 + 0.5^2 = 0.5 < 1.0
        assertTrue(SocketCompatibilityUtils.areEdgeSocketsConnected(socketA, socketB, 1.0));

        // Distance squared: 0.5^2 + 0.5^2 = 0.5 >= 0.2
        assertFalse(SocketCompatibilityUtils.areEdgeSocketsConnected(socketA, socketB, 0.2));
    }
}
