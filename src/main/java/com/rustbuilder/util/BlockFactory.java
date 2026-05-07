package com.rustbuilder.util;

import com.rustbuilder.core.action.BuildAction;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingTier;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.core.DoorType;
import com.rustbuilder.model.core.Orientation;
import com.rustbuilder.model.deployable.LootRoom;
import com.rustbuilder.model.deployable.ToolCupboard;
import com.rustbuilder.model.deployable.Workbench;
import com.rustbuilder.model.structure.Door;
import com.rustbuilder.model.structure.Floor;
import com.rustbuilder.model.structure.Foundation;
import com.rustbuilder.model.structure.TriangleFloor;
import com.rustbuilder.model.structure.TriangleFoundation;
import com.rustbuilder.model.structure.Wall;

/**
 * Factory for constructing and cloning {@link BuildingBlock} instances from
 * {@link BuildAction} parameters.
 *
 * <p>Centralises block-construction logic that was previously duplicated across
 * {@code RLTrainingService.placeBlock()} and {@code RLTrainingService.cloneBlock()}.
 *
 * <p><b>Tech-debt note:</b> {@code BaseGenome.decode()} (L323-368) contains an
 * identical switch-constructor and duplicated {@code tierFromInt/doorTypeFromInt/
 * orientFromInt} helpers. Connecting it to this factory requires a separate audit
 * of the DOORWAY-case semantic difference (auto-Door placement in decode vs
 * explicit Door placement in placeBlock) before unification.
 */
public final class BlockFactory {

    private BlockFactory() {}

    // -------------------------------------------------------------------------
    // Converter helpers (previously duplicated in RLTrainingService and BaseGenome)
    // -------------------------------------------------------------------------

    public static BuildingTier tierFromInt(int t) {
        switch (t) {
            case 0: return BuildingTier.TWIG;
            case 1: return BuildingTier.WOOD;
            case 2: return BuildingTier.STONE;
            case 3: return BuildingTier.METAL;
            case 4: return BuildingTier.HQM;
            default: return BuildingTier.STONE;
        }
    }

    /**
     * Full three-value mapping (matches BaseGenome.doorTypeFromInt).
     * RLTrainingService previously always returned SHEET_METAL; callers that
     * relied on that behaviour can pass {@code 0} explicitly.
     */
    public static DoorType doorTypeFromInt(int d) {
        switch (d) {
            case 0: return DoorType.SHEET_METAL;
            case 1: return DoorType.GARAGE;
            case 2: return DoorType.ARMORED;
            default: return DoorType.SHEET_METAL;
        }
    }

    public static Orientation orientFromInt(int o) {
        switch (o) {
            case 0: return Orientation.NORTH;
            case 1: return Orientation.EAST;
            case 2: return Orientation.SOUTH;
            case 3: return Orientation.WEST;
            default: return Orientation.NORTH;
        }
    }

    // -------------------------------------------------------------------------
    // Block construction
    // -------------------------------------------------------------------------

    /**
     * Creates a new {@link BuildingBlock} for the given action parameters.
     *
     * <p>This mirrors the switch originally in
     * {@code RLTrainingService.placeBlock()}. Placement validation (canPlace,
     * collision, stability) is intentionally <em>not</em> included here — it
     * remains the responsibility of the caller.
     *
     * @param type        action type; must not be {@code null}
     * @param x           world x coordinate
     * @param y           world y coordinate
     * @param z           floor level
     * @param rotation    rotation in degrees
     * @param orientation wall orientation (relevant for Wall/Doorway/WindowFrame)
     * @param doorType    door material type (relevant for Doorway)
     * @return a newly constructed block, or {@code null} for unknown types
     */
    public static BuildingBlock create(
            BuildAction.ActionType type,
            double x, double y, int z,
            double rotation,
            Orientation orientation,
            DoorType doorType) {

        BuildingBlock block = null;

        switch (type) {
            case FOUNDATION:
                block = new Foundation(x, y, z);
                block.setRotation(rotation);
                break;
            case TRIANGLE_FOUNDATION:
                block = new TriangleFoundation(x, y, z, rotation);
                break;
            case WALL:
                block = new Wall(x, y, z, orientation);
                break;
            case DOORWAY: {
                Wall dw = new Wall(x, y, z, orientation);
                dw.setType(BuildingType.DOORWAY);
                dw.setDoorType(doorType);
                block = dw;
                break;
            }
            case WINDOW_FRAME: {
                Wall wf = new Wall(x, y, z, orientation);
                wf.setType(BuildingType.WINDOW_FRAME);
                block = wf;
                break;
            }
            case FLOOR:
                block = new Floor(x, y, z, rotation);
                break;
            case TRIANGLE_FLOOR:
                block = new TriangleFloor(x, y, z, rotation);
                break;
            case TC:
                block = new ToolCupboard(x, y, z, rotation);
                break;
            case WORKBENCH:
                block = new Workbench(x, y, z, rotation);
                break;
            case LOOT_ROOM:
                block = new LootRoom(x, y, z, rotation);
                break;
        }

        return block;
    }

    /**
     * Creates a block directly from the model type used by the manual UI flow.
     */
    public static BuildingBlock create(
            BuildingType type,
            double x, double y, int z,
            double rotation,
            Orientation orientation,
            DoorType doorType) {

        if (type == null) {
            return null;
        }

        switch (type) {
            case FOUNDATION:
                return new Foundation(x, y, z, rotation);
            case TRIANGLE_FOUNDATION:
                return new TriangleFoundation(x, y, z, rotation);
            case WALL:
                return new Wall(x, y, z, orientation);
            case DOORWAY: {
                Wall doorWall = new Wall(x, y, z, orientation);
                doorWall.setType(BuildingType.DOORWAY);
                doorWall.setDoorType(doorType);
                return doorWall;
            }
            case WINDOW_FRAME: {
                Wall windowFrame = new Wall(x, y, z, orientation);
                windowFrame.setType(BuildingType.WINDOW_FRAME);
                return windowFrame;
            }
            case FLOOR:
                return new Floor(x, y, z, rotation);
            case TRIANGLE_FLOOR:
                return new TriangleFloor(x, y, z, rotation);
            case TC:
                return new ToolCupboard(x, y, z, rotation);
            case WORKBENCH:
                return new Workbench(x, y, z, rotation);
            case LOOT_ROOM:
                return new LootRoom(x, y, z, rotation);
            case DOOR:
                return new Door(x, y, z, orientation, doorType);
            default:
                return null;
        }
    }

    // -------------------------------------------------------------------------
    // Block cloning
    // -------------------------------------------------------------------------

    /**
     * Deep-clones a {@link BuildingBlock} through the model-level prototype.
     *
     * @param b the block to clone; may be {@code null}
     * @return a new block with the same type, position, rotation and tier,
     *         or {@code null} if {@code b} is {@code null}
     */
    public static BuildingBlock clone(BuildingBlock b) {
        return b == null ? null : b.clone();
    }
}
