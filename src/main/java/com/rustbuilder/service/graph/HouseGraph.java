package com.rustbuilder.service.graph;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.rustbuilder.config.GameConstants;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.structure.Wall;
import com.rustbuilder.service.evaluator.RaidConstants;
import com.rustbuilder.util.BuildingTypeUtils;

/**
 * Builds a 3D adjacency graph from placed building blocks.
 * Nodes = tile positions (tileX, tileY, floorZ).
 * Edges = connections between adjacent tiles through walls/doors/stairs.
 */
public class HouseGraph {

    /** A node in the house graph, representing a tile position. */
    public static class TileNode {
        public final double x, y;
        public final int z;
        public final NodeKey id;
        public final String type; // "tile", "outside", "tc", "workbench", "loot_room"

        public TileNode(double x, double y, int z, String type) {
            this.x = x;
            this.y = y;
            this.z = z;
            this.type = type;
            this.id = type.equals("outside") ? new NodeKey(0, 0, 0, "OUTSIDE") : new NodeKey(x, y, z, type);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof TileNode)) return false;
            return id.equals(((TileNode) o).id);
        }

        @Override
        public int hashCode() {
            return id.hashCode();
        }

        @Override
        public String toString() {
            return id.toString();
        }
    }

    /** An edge between two tiles, with walk cost and raid cost. */
    public static class TileEdge {
        public final TileNode from, to;
        public final double walkCost;      // steps to walk (∞ if wall blocks)
        public final int raidSulfurCost;   // sulfur to destroy blocking structure
        public final BuildingBlock blocker; // the wall/floor blocking this edge (null if open)

        public TileEdge(TileNode from, TileNode to, double walkCost, int raidSulfurCost, BuildingBlock blocker) {
            this.from = from;
            this.to = to;
            this.walkCost = walkCost;
            this.raidSulfurCost = raidSulfurCost;
            this.blocker = blocker;
        }
    }

    private final Map<NodeKey, TileNode> nodes = new LinkedHashMap<>();
    private final Map<NodeKey, List<TileEdge>> adjacency = new HashMap<>();
    private final TileNode outsideNode = new TileNode(0, 0, 0, "outside");

    public void buildGraph(List<BuildingBlock> blocks) {
        nodes.clear();
        adjacency.clear();

        // Add outside node
        addNode(outsideNode);

        // 1. Identify all foundation/floor tiles and create nodes
        for (BuildingBlock block : blocks) {
            if (isFoundationOrFloor(block)) {
                TileNode node = new TileNode(block.getX(), block.getY(), block.getZ(), "tile");
                addNode(node);
            }
        }

        // 2. Add special deployable nodes on their tile
        for (BuildingBlock block : blocks) {
            if (block.getType() == BuildingType.TC) {
                TileNode node = new TileNode(block.getX(), block.getY(), block.getZ(), "tc");
                addNode(node);
                TileNode tile = findTileNode(block.getX(), block.getY(), block.getZ());
                if (tile != null) {
                    addEdge(node, tile, 0, 0, null);
                    addEdge(tile, node, 0, 0, null);
                }
            } else if (block.getType() == BuildingType.WORKBENCH) {
                TileNode node = new TileNode(block.getX(), block.getY(), block.getZ(), "workbench");
                addNode(node);
                TileNode tile = findTileNode(block.getX(), block.getY(), block.getZ());
                if (tile != null) {
                    addEdge(node, tile, 0, 0, null);
                    addEdge(tile, node, 0, 0, null);
                }
            } else if (block.getType() == BuildingType.LOOT_ROOM) {
                TileNode node = new TileNode(block.getX(), block.getY(), block.getZ(), "loot_room");
                addNode(node);
                TileNode tile = findTileNode(block.getX(), block.getY(), block.getZ());
                if (tile != null) {
                    addEdge(node, tile, 0, 0, null);
                    addEdge(tile, node, 0, 0, null);
                }
            }
        }

        // --- Index structures built once, used by steps 3-6 ---

        // Flat list of tile nodes (no special/outside nodes)
        List<TileNode> tileNodes = new ArrayList<>();
        for (TileNode n : nodes.values()) {
            if ("tile".equals(n.type)) tileNodes.add(n);
        }

        // tileIndex: O(1) lookup by position — used to find cardinal neighbours
        Map<NodeKey, TileNode> tileIndex = new HashMap<>();
        for (TileNode n : tileNodes) {
            tileIndex.put(n.id, n);
        }

        // wallsByFloor: pre-filter walls per floor so findWallBetween doesn't scan all blocks
        Map<Integer, List<BuildingBlock>> wallsByFloor = new HashMap<>();
        for (BuildingBlock b : blocks) {
            if (isWallType(b)) {
                wallsByFloor.computeIfAbsent(b.getZ(), k -> new ArrayList<>()).add(b);
            }
        }

        // 3. Build horizontal edges between adjacent tiles (same floor)
        //    For square tiles: only 4 cardinal offsets (E, W, N, S).
        //    For triangle tiles (fewer vertices): fall back to hypot check.
        double T = GameConstants.TILE_SIZE;

        for (TileNode a : tileNodes) {
            // Determine if this tile is a triangle (3 polygon vertices)
            BuildingBlock aBlock = getBlockForTile(blocks, a);
            boolean aIsTriangle = aBlock != null &&
                    (aBlock.getType() == BuildingType.TRIANGLE_FOUNDATION ||
                     aBlock.getType() == BuildingType.TRIANGLE_FLOOR);

            if (!aIsTriangle) {
                // Square tile: indexed 4-direction lookup.
                // Process only the two "forward" directions (E and S) per node so each
                // undirected pair is handled exactly once; addHorizontalEdges adds both directions.
                double[] fwdDx = { T, 0 };
                double[] fwdDy = { 0, T };
                for (int d = 0; d < 2; d++) {
                    double nx = a.x + fwdDx[d];
                    double ny = a.y + fwdDy[d];
                    NodeKey nk = new NodeKey(nx, ny, a.z, "tile");
                    TileNode b = tileIndex.get(nk);
                    if (b == null) continue;

                    List<BuildingBlock> floorWalls = wallsByFloor.getOrDefault(a.z, java.util.Collections.emptyList());
                    BuildingBlock wall = findWallBetween(floorWalls, a, b);
                    addHorizontalEdges(a, b, wall);
                }
            } else {
                // Triangle tile: fallback pairwise check within this tile's floor
                for (TileNode b : tileNodes) {
                    if (b == a || b.z != a.z) continue;
                    if (a.id.hashCode() >= b.id.hashCode()) continue; // process once
                    double dist = Math.hypot(a.x - b.x, a.y - b.y);
                    if (dist < T * 1.1) {
                        List<BuildingBlock> floorWalls = wallsByFloor.getOrDefault(a.z, java.util.Collections.emptyList());
                        BuildingBlock wall = findWallBetween(floorWalls, a, b);
                        addHorizontalEdges(a, b, wall);
                    }
                }
            }
        }

        // 4. Vertical edges (stairs between floors)
        for (BuildingBlock block : blocks) {
            if (block.getType() == BuildingType.STAIRS) {
                TileNode below = findTileNode(block.getX(), block.getY(), block.getZ());
                TileNode above = findTileNode(block.getX(), block.getY(), block.getZ() + 1);
                if (below != null && above != null) {
                    addEdge(below, above, 3, 0, null);
                    addEdge(above, below, 3, 0, null);
                }
            }
        }

        // 5. Connect outside to tiles that have an exposed edge (no wall on boundary)
        for (TileNode tile : tileNodes) {
            if (hasExposedEdge(blocks, tile, wallsByFloor, tileIndex)) {
                addEdge(outsideNode, tile, 1, 0, null);
                addEdge(tile, outsideNode, 1, 0, null);
            }
        }

        // 6. Outside connects through outer walls for raiding
        for (TileNode tile : tileNodes) {
            List<BuildingBlock> outerWalls = findOuterWalls(blocks, tile, wallsByFloor, tileIndex);
            for (BuildingBlock wall : outerWalls) {
                int sulfur;
                double walkCost;
                if (wall.getType() == BuildingType.DOORWAY) {
                    int doorSulfur = (wall instanceof Wall) ? ((Wall) wall).getDoorType().getSulfurCost() : 0;
                    int frameSulfur = RaidConstants.getWallSulfurCost(wall.getTier());
                    sulfur = Math.min(doorSulfur, frameSulfur);
                    walkCost = 2;
                } else {
                    sulfur = RaidConstants.getWallSulfurCost(wall.getTier());
                    walkCost = Double.MAX_VALUE;
                }
                addEdge(outsideNode, tile, walkCost, sulfur, wall);
                addEdge(tile, outsideNode, walkCost, sulfur, wall);
            }
        }

        // 7. Vertical raid edges through floors/ceilings
        for (BuildingBlock block : blocks) {
            if (isFloor(block) && block.getZ() > 0) {
                TileNode below = findTileNode(block.getX(), block.getY(), block.getZ() - 1);
                TileNode above = findTileNode(block.getX(), block.getY(), block.getZ());
                if (below != null && above != null) {
                    int sulfur = RaidConstants.getWallSulfurCost(block.getTier());
                    addEdge(below, above, Double.MAX_VALUE, sulfur, block);
                    addEdge(above, below, Double.MAX_VALUE, sulfur, block);
                }
            }
        }
    }

    /** Shared edge-creation for horizontal pairs, extracted to avoid code duplication. */
    private void addHorizontalEdges(TileNode a, TileNode b, BuildingBlock wall) {
        if (wall == null) {
            addEdge(a, b, 1, 0, null);
            addEdge(b, a, 1, 0, null);
        } else if (wall.getType() == BuildingType.DOORWAY) {
            int doorSulfur = (wall instanceof Wall) ? ((Wall) wall).getDoorType().getSulfurCost() : 0;
            int frameSulfur = RaidConstants.getWallSulfurCost(wall.getTier());
            addEdge(a, b, 2, Math.min(doorSulfur, frameSulfur), wall);
            addEdge(b, a, 2, Math.min(doorSulfur, frameSulfur), wall);
        } else {
            int sulfur = RaidConstants.getWallSulfurCost(wall.getTier());
            addEdge(a, b, Double.MAX_VALUE, sulfur, wall);
            addEdge(b, a, Double.MAX_VALUE, sulfur, wall);
        }
    }

    // === Graph Query Methods ===

    public TileNode getOutsideNode() {
        return outsideNode;
    }

    public Collection<TileNode> getAllNodes() {
        return nodes.values();
    }

    public List<TileEdge> getEdges(TileNode node) {
        return adjacency.getOrDefault(node.id, Collections.emptyList());
    }

    public TileNode findNodeByType(String type) {
        for (TileNode n : nodes.values()) {
            if (n.type.equals(type)) return n;
        }
        return null;
    }

    public List<TileNode> findAllNodesByType(String type) {
        List<TileNode> result = new ArrayList<>();
        for (TileNode n : nodes.values()) {
            if (n.type.equals(type)) result.add(n);
        }
        return result;
    }

    /**
     * Returns all edges in the graph (for splash damage analysis).
     */
    public List<TileEdge> getAllEdges() {
        List<TileEdge> all = new ArrayList<>();
        for (List<TileEdge> edges : adjacency.values()) {
            all.addAll(edges);
        }
        return all;
    }

    // === Internal Helpers ===

    private void addNode(TileNode node) {
        if (!nodes.containsKey(node.id)) {
            nodes.put(node.id, node);
            adjacency.put(node.id, new ArrayList<>());
        }
    }

    private void addEdge(TileNode from, TileNode to, double walkCost, int raidSulfur, BuildingBlock blocker) {
        adjacency.computeIfAbsent(from.id, k -> new ArrayList<>())
                  .add(new TileEdge(from, to, walkCost, raidSulfur, blocker));
    }

    private TileNode findTileNode(double x, double y, int z) {
        NodeKey id = new NodeKey(x, y, z, "tile");
        return nodes.get(id);
    }

    private boolean isFoundationOrFloor(BuildingBlock b) {
        return BuildingTypeUtils.isHorizontalSurface(b.getType());
    }

    private boolean isFloor(BuildingBlock b) {
        return BuildingTypeUtils.isFloor(b.getType());
    }

    private boolean isWallType(BuildingBlock b) {
        return BuildingTypeUtils.isWall(b.getType());
    }

    /**
     * Find wall between two adjacent tiles using a pre-filtered wall list for their floor.
     */
    private BuildingBlock findWallBetween(List<BuildingBlock> floorWalls, TileNode a, TileNode b) {
        // Tile centers
        double ax = a.x + GameConstants.HALF_TILE;
        double ay = a.y + GameConstants.HALF_TILE;
        double bx = b.x + GameConstants.HALF_TILE;
        double by = b.y + GameConstants.HALF_TILE;

        for (BuildingBlock block : floorWalls) {
            if (!isWallType(block)) continue;
            // floor already filtered by caller

            double[] poly = block.getCollisionPoints();
            if (poly == null || poly.length < 8) continue;

            // Check if the line segment A->B intersects any of the 4 edges of the wall's polygon
            for (int i = 0; i < 4; i++) {
                double x1 = poly[i * 2];
                double y1 = poly[i * 2 + 1];
                double x2 = poly[((i + 1) % 4) * 2];
                double y2 = poly[((i + 1) % 4) * 2 + 1];

                if (lineIntersectsLine(ax, ay, bx, by, x1, y1, x2, y2)) {
                    return block;
                }
            }
        }
        return null;
    }

    /**
     * Standard line segment intersection algorithm.
     */
    private boolean lineIntersectsLine(double x1, double y1, double x2, double y2,
                                       double x3, double y3, double x4, double y4) {
        double d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
        if (Math.abs(d) < 1e-6) return false;

        double t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d;
        double u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / d;

        return t >= -0.05 && t <= 1.05 && u >= -0.05 && u <= 1.05;
    }


    private BuildingBlock getBlockForTile(List<BuildingBlock> blocks, TileNode tile) {
        for (BuildingBlock b : blocks) {
            if (isFoundationOrFloor(b) && b.getZ() == tile.z && 
                Math.abs(b.getX() - tile.x) < 0.1 && Math.abs(b.getY() - tile.y) < 0.1) {
                return b;
            }
        }
        return null;
    }

    private List<double[]> getOuterCheckPoints(BuildingBlock tileBlock) {
        List<double[]> checkPoints = new ArrayList<>();
        double[] poly = tileBlock.getCollisionPoints();
        if (poly == null || poly.length < 6) return checkPoints;

        int n = poly.length / 2;
        double cx = tileBlock.getX() + GameConstants.HALF_TILE;
        double cy = tileBlock.getY() + GameConstants.HALF_TILE;
        
        for (int i = 0; i < n; i++) {
            double x1 = poly[i * 2];
            double y1 = poly[i * 2 + 1];
            double x2 = poly[((i + 1) % n) * 2];
            double y2 = poly[((i + 1) % n) * 2 + 1];
            
            double mx = (x1 + x2) / 2.0;
            double my = (y1 + y2) / 2.0;
            
            double dx = mx - cx;
            double dy = my - cy;
            double len = Math.hypot(dx, dy);
            if (len > 0) { dx /= len; dy /= len; }
            
            checkPoints.add(new double[] { cx + dx * 60.0, cy + dy * 60.0 });
        }
        return checkPoints;
    }

    private boolean hasExposedEdge(List<BuildingBlock> blocks, TileNode tile,
                                   Map<Integer, List<BuildingBlock>> wallsByFloor,
                                   Map<NodeKey, TileNode> tileIndex) {
        BuildingBlock tileBlock = getBlockForTile(blocks, tile);
        if (tileBlock == null) return false;

        List<double[]> checkPoints = getOuterCheckPoints(tileBlock);

        for (double[] pt : checkPoints) {
            // Compute candidate neighbour tile top-left from the check point
            double candidateX = Math.round((pt[0] - GameConstants.HALF_TILE) / GameConstants.TILE_SIZE) * GameConstants.TILE_SIZE;
            double candidateY = Math.round((pt[1] - GameConstants.HALF_TILE) / GameConstants.TILE_SIZE) * GameConstants.TILE_SIZE;
            NodeKey candidateKey = new NodeKey(candidateX, candidateY, tile.z, "tile");
            boolean hasNeighbor = tileIndex.containsKey(candidateKey);

            if (!hasNeighbor) {
                TileNode tempNeighbor = new TileNode(pt[0] - GameConstants.HALF_TILE, pt[1] - GameConstants.HALF_TILE, tile.z, "tile");
                List<BuildingBlock> floorWalls = wallsByFloor.getOrDefault(tile.z, java.util.Collections.emptyList());
                BuildingBlock wall = findWallBetween(floorWalls, tile, tempNeighbor);
                if (wall == null) return true;
            }
        }
        return false;
    }

    private List<BuildingBlock> findOuterWalls(List<BuildingBlock> blocks, TileNode tile,
                                               Map<Integer, List<BuildingBlock>> wallsByFloor,
                                               Map<NodeKey, TileNode> tileIndex) {
        List<BuildingBlock> outerWalls = new ArrayList<>();
        BuildingBlock tileBlock = getBlockForTile(blocks, tile);
        if (tileBlock == null) return outerWalls;

        List<double[]> checkPoints = getOuterCheckPoints(tileBlock);

        for (double[] pt : checkPoints) {
            double candidateX = Math.round((pt[0] - GameConstants.HALF_TILE) / GameConstants.TILE_SIZE) * GameConstants.TILE_SIZE;
            double candidateY = Math.round((pt[1] - GameConstants.HALF_TILE) / GameConstants.TILE_SIZE) * GameConstants.TILE_SIZE;
            NodeKey candidateKey = new NodeKey(candidateX, candidateY, tile.z, "tile");
            boolean hasNeighbor = tileIndex.containsKey(candidateKey);

            if (!hasNeighbor) {
                TileNode tempNeighbor = new TileNode(pt[0] - GameConstants.HALF_TILE, pt[1] - GameConstants.HALF_TILE, tile.z, "tile");
                List<BuildingBlock> floorWalls = wallsByFloor.getOrDefault(tile.z, java.util.Collections.emptyList());
                BuildingBlock wall = findWallBetween(floorWalls, tile, tempNeighbor);
                if (wall != null) outerWalls.add(wall);
            }
        }
        return outerWalls;
    }
}
