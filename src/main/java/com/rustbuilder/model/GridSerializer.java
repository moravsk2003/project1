package com.rustbuilder.model;

import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingTier;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.core.DoorType;
import com.rustbuilder.model.core.Orientation;
import com.rustbuilder.util.BlockFactory;
import com.rustbuilder.model.structure.Wall;
import com.rustbuilder.model.structure.Door;

import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.Locale;
import java.util.logging.Level;
import java.util.logging.Logger;

public class GridSerializer {
    private static final Logger LOGGER = Logger.getLogger(GridSerializer.class.getName());

    public static String toJson(GridModel grid) {
        if (grid == null) return "[]";
        StringBuilder sb = new StringBuilder();
        sb.append("[\n");
        boolean first = true;
        for (BuildingBlock b : grid.getAllBlocks()) {
            if (!first) sb.append(",\n");
            
            Orientation ori = Orientation.NORTH;
            DoorType dt = DoorType.SHEET_METAL;
            if (b instanceof Wall) {
                ori = ((Wall) b).getOrientation();
                if (((Wall) b).getDoorType() != null) {
                    dt = ((Wall) b).getDoorType();
                }
            } else if (b instanceof Door) {
                ori = ((Door) b).getOrientation();
                dt = ((Door) b).getDoorType();
            }

            sb.append(String.format(Locale.US,
                "  {\"type\":\"%s\", \"tier\":\"%s\", \"x\":%.2f, \"y\":%.2f, \"z\":%d, \"rot\":%.2f, \"ori\":\"%s\", \"door\":\"%s\"}", 
                b.getType().name(), b.getTier().name(), b.getX(), b.getY(), b.getZ(), b.getRotation(), ori.name(), dt.name()));
            first = false;
        }
        sb.append("\n]");
        return sb.toString();
    }

    public static GridModel fromJson(String json) {
        GridModel grid = new GridModel();
        if (json == null || json.isBlank() || json.equals("[]")) {
            return grid;
        }

        Pattern p = Pattern.compile("\\{\"type\":\"([A-Z_]+)\", \"tier\":\"([A-Z_]+)\", \"x\":([\\-\\d\\.]+), \"y\":([\\-\\d\\.]+), \"z\":([\\-\\d]+), \"rot\":([\\-\\d\\.]+), \"ori\":\"([A-Z_]+)\", \"door\":\"([A-Z_]+)\"\\}");
        Matcher m = p.matcher(json);

        while (m.find()) {
            try {
                BuildingType type = BuildingType.valueOf(m.group(1));
                BuildingTier tier = BuildingTier.valueOf(m.group(2));
                double x = Double.parseDouble(m.group(3));
                double y = Double.parseDouble(m.group(4));
                int z = Integer.parseInt(m.group(5));
                double rot = Double.parseDouble(m.group(6));
                Orientation ori = Orientation.valueOf(m.group(7));
                DoorType dt = DoorType.valueOf(m.group(8));

                BuildingBlock b = BlockFactory.create(type, x, y, z, rot, ori, dt);
                if (b != null) {
                    b.setTier(tier);
                    grid.addBlockSilent(b);
                }
            } catch (IllegalArgumentException e) {
                LOGGER.log(Level.FINE, "Skipping malformed serialized block.", e);
            }
        }
        grid.finalizeLoad();
        return grid;
    }
}
