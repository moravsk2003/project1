package com.rustbuilder.model.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.structure.Door;
import com.rustbuilder.model.structure.Foundation;
import com.rustbuilder.model.structure.Wall;

class BuildingBlockPrototypeTest {

    @Test
    void clone_preservesCommonStateAndWallSpecificState() {
        Wall doorway = new Wall(10, 20, 2, Orientation.EAST);
        doorway.setType(BuildingType.DOORWAY);
        doorway.setDoorType(DoorType.GARAGE);
        doorway.setTier(BuildingTier.METAL);
        doorway.setRotation(90);
        doorway.setStability(0.72);

        BuildingBlock clone = doorway.clone();

        assertNotSame(doorway, clone);
        assertNotEquals(doorway.getId(), clone.getId());
        assertTrue(clone instanceof Wall);
        Wall clonedWall = (Wall) clone;
        assertEquals(BuildingType.DOORWAY, clonedWall.getType());
        assertEquals(Orientation.EAST, clonedWall.getOrientation());
        assertEquals(DoorType.GARAGE, clonedWall.getDoorType());
        assertEquals(BuildingTier.METAL, clonedWall.getTier());
        assertEquals(90, clonedWall.getRotation());
        assertEquals(0.72, clonedWall.getStability());
    }

    @Test
    void clone_preservesDoorSpecificState() {
        Door door = new Door(30, 40, 1, Orientation.SOUTH, DoorType.ARMORED);
        door.setRotation(180);
        door.setStability(1.0);

        BuildingBlock clone = door.clone();

        assertNotSame(door, clone);
        assertTrue(clone instanceof Door);
        Door clonedDoor = (Door) clone;
        assertEquals(BuildingType.DOOR, clonedDoor.getType());
        assertEquals(Orientation.SOUTH, clonedDoor.getOrientation());
        assertEquals(DoorType.ARMORED, clonedDoor.getDoorType());
        assertEquals(180, clonedDoor.getRotation());
        assertEquals(1.0, clonedDoor.getStability());
    }

    @Test
    void gridModelClone_usesPrototypeCopies() {
        GridModel grid = new GridModel();
        Foundation foundation = new Foundation(0, 0, 0, 45);
        foundation.setTier(BuildingTier.STONE);
        grid.addBlockSilent(foundation);

        GridModel clone = grid.clone();
        BuildingBlock clonedBlock = clone.getAllBlocks().get(0);

        assertNotSame(foundation, clonedBlock);
        assertEquals(BuildingType.FOUNDATION, clonedBlock.getType());
        assertEquals(BuildingTier.STONE, clonedBlock.getTier());
        assertEquals(45, clonedBlock.getRotation());
    }
}
