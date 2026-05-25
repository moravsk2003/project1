package com.rustbuilder.util;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.Test;

public class CollisionUtilsTest {

    @Test
    void testCheckCollisionWithIntersectingPolygons() {
        // Square 1: [0,0] to [10,10]
        double[] poly1 = {
            0, 0,
            10, 0,
            10, 10,
            0, 10
        };

        // Square 2: [5,5] to [15,15] (Overlapping)
        double[] poly2 = {
            5, 5,
            15, 5,
            15, 15,
            5, 15
        };

        assertTrue(CollisionUtils.checkCollision(poly1, poly2));
    }

    @Test
    void testCheckCollisionWithSeparatedPolygons() {
        // Square 1: [0,0] to [10,10]
        double[] poly1 = {
            0, 0,
            10, 0,
            10, 10,
            0, 10
        };

        // Square 2: [12,0] to [22,10] (Separated on X axis)
        double[] poly2 = {
            12, 0,
            22, 0,
            22, 10,
            12, 10
        };

        assertFalse(CollisionUtils.checkCollision(poly1, poly2));
    }

    @Test
    void testCheckCollisionWithEmptyPolygons() {
        double[] poly1 = {};
        double[] poly2 = {
            0, 0,
            1, 0,
            1, 1,
            0, 1
        };

        assertFalse(CollisionUtils.checkCollision(poly1, poly2));
        assertFalse(CollisionUtils.checkCollision(poly2, poly1));
    }

    @Test
    void testCheckCollisionTouchingEdges() {
        // Square 1: [0,0] to [10,10]
        double[] poly1 = {
            0, 0,
            10, 0,
            10, 10,
            0, 10
        };

        // Square 2: [10,0] to [20,10] (Touching exactly at X=10)
        double[] poly2 = {
            10, 0,
            20, 0,
            20, 10,
            10, 10
        };

        // touching polygons are considered separated because of maxA <= minB + EPSILON
        assertFalse(CollisionUtils.checkCollision(poly1, poly2));
    }
}
