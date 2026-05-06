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

    // -------------------------------------------------------------------------
    // Block cloning
    // -------------------------------------------------------------------------

    /**
     * Deep-clones a {@link BuildingBlock}, preserving rotation and tier.
     *
     * <p>This mirrors the instanceof-chain originally in
     * {@code RLTrainingService.cloneBlock()}.
     *
     * @param b the block to clone; may be {@code null}
     * @return a new block with the same type, position, rotation and tier,
     *         or {@code null} if {@code b} is {@code null} or its type is unrecognised
     */
    public static BuildingBlock clone(BuildingBlock b) {
        if (b == null) return null;

        BuildingBlock clone = null;

        if (b instanceof Foundation) {
            clone = new Foundation(b.getX(), b.getY(), b.getZ());
        } else if (b instanceof TriangleFoundation) {
            clone = new TriangleFoundation(b.getX(), b.getY(), b.getZ(), b.getRotation());
        } else if (b instanceof Wall) {
            Wall w = (Wall) b;
            Wall wClone = new Wall(w.getX(), w.getY(), w.getZ(), w.getOrientation());
            wClone.setType(w.getType());
            if (w.getType() == BuildingType.DOORWAY) {
                wClone.setDoorType(w.getDoorType());
            }
            clone = wClone;
        } else if (b instanceof Floor) {
            clone = new Floor(b.getX(), b.getY(), b.getZ(), b.getRotation());
        } else if (b instanceof TriangleFloor) {
            clone = new TriangleFloor(b.getX(), b.getY(), b.getZ(), b.getRotation());
        } else if (b instanceof Door) {
            Door d = (Door) b;
            clone = new Door(d.getX(), d.getY(), d.getZ(), d.getOrientation(), d.getDoorType());
        } else if (b instanceof ToolCupboard) {
            clone = new ToolCupboard(b.getX(), b.getY(), b.getZ(), b.getRotation());
        } else if (b instanceof Workbench) {
            clone = new Workbench(b.getX(), b.getY(), b.getZ(), b.getRotation());
        } else if (b instanceof LootRoom) {
            clone = new LootRoom(b.getX(), b.getY(), b.getZ(), b.getRotation());
        }

        if (clone != null) {
            clone.setRotation(b.getRotation());
            clone.setTier(b.getTier());
        }
        return clone;
    }
}
