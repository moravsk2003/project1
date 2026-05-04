package com.rustbuilder.service.evaluator;

import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.service.graph.HouseGraph;
import com.rustbuilder.service.graph.HouseGraph.TileEdge;
import com.rustbuilder.service.graph.HouseGraph.TileNode;
import com.rustbuilder.service.graph.NodeKey;
import com.rustbuilder.util.BuildingTypeUtils;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Evaluates working area using the house graph:
 * raw = protected tile area / protected component perimeter.
 */
public class WorkingAreaEvaluator {

    private static final double TRIANGLE_AREA = 0.4330127019;
    private static final double EPS = 1e-9;

    public static class WorkingAreaResult {
        public final double protectedArea;
        public final int openEdgePerimeter;
        public final int protectedTiles;
        public final double rawScore;
        public final double score;

        public WorkingAreaResult(double protectedArea, int openEdgePerimeter, int protectedTiles, double rawScore, double score) {
            this.protectedArea = protectedArea;
            this.openEdgePerimeter = openEdgePerimeter;
            this.protectedTiles = protectedTiles;
            this.rawScore = rawScore;
            this.score = score;
        }

        @Override
        public String toString() {
            return String.format(
                "Working Area: area=%.2f, perimeter=%d, tiles=%d, raw=%.3f, score=%.3f",
                protectedArea, openEdgePerimeter, protectedTiles, rawScore, score
            );
        }
    }

    public WorkingAreaResult evaluate(HouseGraph graph, List<BuildingBlock> blocks) {
        Map<NodeKey, Double> tileAreas = buildTileAreaMap(blocks);
        Set<TileNode> protectedTiles = findProtectedTiles(graph);
        if (protectedTiles.isEmpty()) {
            return new WorkingAreaResult(0.0, 0, 0, 0.0, 0.0);
        }

        Set<TileNode> selectedComponent = selectWorkingComponent(graph, protectedTiles);
        if (selectedComponent.isEmpty()) {
            return new WorkingAreaResult(0.0, 0, 0, 0.0, 0.0);
        }

        double area = 0.0;
        for (TileNode tile : selectedComponent) {
            area += tileAreas.getOrDefault(tile.id, 1.0);
        }

        int openPerimeter = countOpenPerimeterEdges(graph, selectedComponent);
        double raw = openPerimeter > 0 ? area / openPerimeter : 0.0;
        double score = normalize(raw);

        return new WorkingAreaResult(area, openPerimeter, selectedComponent.size(), raw, score);
    }

    private Map<NodeKey, Double> buildTileAreaMap(List<BuildingBlock> blocks) {
        Map<NodeKey, Double> areas = new HashMap<>();
        for (BuildingBlock block : blocks) {
            BuildingType type = block.getType();
            if (!BuildingTypeUtils.isHorizontalSurface(type)) {
                continue;
            }

            double area = tileArea(type);
            NodeKey key = new NodeKey(block.getX(), block.getY(), block.getZ(), "tile");
            Double existing = areas.get(key);
            if (existing == null || area > existing) {
                areas.put(key, area);
            }
        }
        return areas;
    }

    private double tileArea(BuildingType type) {
        if (type == BuildingType.TRIANGLE_FOUNDATION || type == BuildingType.TRIANGLE_FLOOR) {
            return TRIANGLE_AREA;
        }
        return 1.0;
    }

    private Set<TileNode> findProtectedTiles(HouseGraph graph) {
        TileNode outside = graph.getOutsideNode();
        Set<TileNode> reachable = new HashSet<>();
        ArrayDeque<TileNode> queue = new ArrayDeque<>();
        queue.add(outside);
        reachable.add(outside);

        while (!queue.isEmpty()) {
            TileNode cur = queue.poll();
            Set<NodeKey> seenTargets = new HashSet<>();
            for (TileEdge edge : graph.getEdges(cur)) {
                if (!isPenetrableFromOutside(edge)) {
                    continue;
                }
                TileNode next = edge.to;
                if (!seenTargets.add(next.id)) {
                    continue;
                }
                if (reachable.add(next)) {
                    queue.add(next);
                }
            }
        }

        Set<TileNode> protectedTiles = new HashSet<>();
        for (TileNode node : graph.getAllNodes()) {
            if ("tile".equals(node.type) && !reachable.contains(node)) {
                protectedTiles.add(node);
            }
        }
        return protectedTiles;
    }

    private Set<TileNode> selectWorkingComponent(HouseGraph graph, Set<TileNode> protectedTiles) {
        List<Set<TileNode>> components = splitIntoComponents(graph, protectedTiles);
        if (components.isEmpty()) {
            return Set.of();
        }

        Set<TileNode> tcComponent = findTcComponent(graph, components);
        if (tcComponent != null) {
            return tcComponent;
        }

        Set<TileNode> largest = components.get(0);
        for (Set<TileNode> comp : components) {
            if (comp.size() > largest.size()) {
                largest = comp;
            }
        }
        return largest;
    }

    private Set<TileNode> findTcComponent(HouseGraph graph, List<Set<TileNode>> components) {
        TileNode tcNode = graph.findNodeByType("tc");
        if (tcNode == null) {
            return null;
        }

        TileNode tcTile = null;
        for (TileEdge edge : graph.getEdges(tcNode)) {
            if ("tile".equals(edge.to.type)) {
                tcTile = edge.to;
                break;
            }
        }
        if (tcTile == null) {
            return null;
        }

        for (Set<TileNode> comp : components) {
            if (comp.contains(tcTile)) {
                return comp;
            }
        }
        return null;
    }

    private List<Set<TileNode>> splitIntoComponents(HouseGraph graph, Set<TileNode> protectedTiles) {
        Set<TileNode> visited = new HashSet<>();
        List<Set<TileNode>> components = new ArrayList<>();

        for (TileNode start : protectedTiles) {
            if (visited.contains(start)) {
                continue;
            }

            Set<TileNode> component = new HashSet<>();
            ArrayDeque<TileNode> queue = new ArrayDeque<>();
            queue.add(start);
            visited.add(start);

            while (!queue.isEmpty()) {
                TileNode cur = queue.poll();
                component.add(cur);

                Set<NodeKey> seenTargets = new HashSet<>();
                for (TileEdge edge : graph.getEdges(cur)) {
                    if (!isWalkable(edge) || !"tile".equals(edge.to.type)) {
                        continue;
                    }

                    TileNode next = edge.to;
                    if (!seenTargets.add(next.id)) {
                        continue;
                    }
                    if (!protectedTiles.contains(next) || visited.contains(next)) {
                        continue;
                    }

                    visited.add(next);
                    queue.add(next);
                }
            }

            components.add(component);
        }

        return components;
    }

    private int countOpenPerimeterEdges(HouseGraph graph, Set<TileNode> component) {
        int perimeter = 0;
        for (TileNode tile : component) {
            Set<String> seenBoundaryTargets = new HashSet<>();
            for (TileEdge edge : graph.getEdges(tile)) {
                TileNode next = edge.to;
                if ("tile".equals(next.type)) {
                    String key = next.id.toString();
                    if (!seenBoundaryTargets.add(key)) {
                        continue;
                    }
                    if (!component.contains(next)) {
                        perimeter++;
                    }
                } else if ("outside".equals(next.type)) {
                    String key = outsideBoundaryKey(edge);
                    if (!seenBoundaryTargets.add(key)) {
                        continue;
                    }
                    perimeter++;
                }
            }
        }
        return perimeter;
    }

    private String outsideBoundaryKey(TileEdge edge) {
        if (edge.blocker != null) {
            return "outside:" + edge.blocker.getId();
        }
        return "outside:open";
    }

    /**
     * For working-area protection, outside should only penetrate through truly open passages.
     * Edges that are blocked by structures (walls/doorways/window frames/floors) are not
     * considered penetrable even if other evaluators treat them as traversable for heuristics.
     */
    private boolean isPenetrableFromOutside(TileEdge edge) {
        return isWalkable(edge) && edge.blocker == null;
    }

    private boolean isWalkable(TileEdge edge) {
        return edge.walkCost < Double.MAX_VALUE - EPS;
    }

    private double normalize(double raw) {
        if (raw <= 0) {
            return 0.0;
        }
        return raw / (1.0 + raw);
    }
}
