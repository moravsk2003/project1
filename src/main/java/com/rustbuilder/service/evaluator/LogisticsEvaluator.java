package com.rustbuilder.service.evaluator;

import com.rustbuilder.service.graph.HouseGraph;
import com.rustbuilder.service.graph.HouseGraph.TileEdge;
import com.rustbuilder.service.graph.HouseGraph.TileNode;
import com.rustbuilder.service.graph.NodeKey;
import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * Evaluates protected logistics between key base locations.
 */
public class LogisticsEvaluator {

    public static class LogisticsResult {
        public final double entranceToTC;
        public final double entranceToWorkbench;
        public final double entranceToLootRoom;
        public final double tcToLootRoom;
        public final double score; // 0.0 - 1.0, higher = better logistics

        public LogisticsResult(double entranceToTC, double entranceToWorkbench,
                               double entranceToLootRoom, double tcToLootRoom, double score) {
            this.entranceToTC = entranceToTC;
            this.entranceToWorkbench = entranceToWorkbench;
            this.entranceToLootRoom = entranceToLootRoom;
            this.tcToLootRoom = tcToLootRoom;
            this.score = score;
        }

        @Override
        public String toString() {
            return String.format(
                "Logistics: Outside->TC=%.1f, TC->WB=%.1f, TC->LR=%.1f | Score=%.2f",
                entranceToTC, entranceToWorkbench, tcToLootRoom, score
            );
        }
    }

    /**
     * Outside->TC must use a protected entrance. TC->WB/LR routes only count
     * when the path stays inside the base and the target is not open to outside.
     */
    public LogisticsResult evaluate(HouseGraph graph) {
        TileNode outside = graph.getOutsideNode();
        TileNode tc = graph.findNodeByType("tc");

        if (tc == null) {
            return emptyResult();
        }

        if (isOpenToOutside(graph, tc)) {
            return new LogisticsResult(0.0, Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE, 0.0);
        }

        Map<NodeKey, Double> distFromOutside = dijkstraWalk(graph, outside, false);
        double dOutsideToTC = distFromOutside.getOrDefault(tc.id, Double.MAX_VALUE);
        if (!isFinitePositive(dOutsideToTC)) {
            return new LogisticsResult(dOutsideToTC, Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE, 0.0);
        }

        Map<NodeKey, Double> protectedDistFromTC = dijkstraWalk(graph, tc, true);
        double dTCtoWB = nearestProtectedDistance(graph, protectedDistFromTC, "workbench");
        double dTCtoLR = nearestProtectedDistance(graph, protectedDistFromTC, "loot_room");

        double totalDist = dOutsideToTC;
        int pathCount = 1;

        if (isFinitePositive(dTCtoWB)) {
            totalDist += dTCtoWB;
            pathCount++;
        }
        if (isFinitePositive(dTCtoLR)) {
            totalDist += dTCtoLR;
            pathCount++;
        }

        double avgDist = totalDist / pathCount;
        double score = 1.0 / (1.0 + avgDist / 10.0);

        return new LogisticsResult(dOutsideToTC, dTCtoWB, dTCtoLR, dTCtoLR, score);
    }

    private LogisticsResult emptyResult() {
        return new LogisticsResult(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE, 0.0);
    }

    private static class DistNode implements Comparable<DistNode> {
        final NodeKey id;
        final double dist;

        DistNode(NodeKey id, double dist) {
            this.id = id;
            this.dist = dist;
        }

        @Override
        public int compareTo(DistNode o) {
            return Double.compare(this.dist, o.dist);
        }
    }

    private Map<NodeKey, Double> dijkstraWalk(HouseGraph graph, TileNode start, boolean rejectOutsideNode) {
        Map<NodeKey, Double> dist = new HashMap<>();
        dist.put(start.id, 0.0);

        Map<NodeKey, TileNode> nodeMap = new HashMap<>();
        for (TileNode n : graph.getAllNodes()) {
            nodeMap.put(n.id, n);
        }

        Set<NodeKey> visited = new HashSet<>();
        PriorityQueue<DistNode> queue = new PriorityQueue<>();
        queue.add(new DistNode(start.id, 0.0));

        while (!queue.isEmpty()) {
            DistNode current = queue.poll();
            if (visited.contains(current.id)) {
                continue;
            }
            visited.add(current.id);

            TileNode curNode = nodeMap.get(current.id);
            if (curNode == null) {
                continue;
            }

            for (TileEdge edge : graph.getEdges(curNode)) {
                if (edge.walkCost >= Double.MAX_VALUE / 2) {
                    continue;
                }
                if (rejectOutsideNode && "outside".equals(edge.to.type)) {
                    continue;
                }

                double newDist = current.dist + edge.walkCost;
                NodeKey neighborId = edge.to.id;
                if (newDist < dist.getOrDefault(neighborId, Double.MAX_VALUE)) {
                    dist.put(neighborId, newDist);
                    queue.add(new DistNode(neighborId, newDist));
                }
            }
        }

        return dist;
    }

    private double nearestProtectedDistance(HouseGraph graph, Map<NodeKey, Double> distFromTC, String nodeType) {
        List<TileNode> nodes = graph.findAllNodesByType(nodeType);
        double best = Double.MAX_VALUE;
        for (TileNode node : nodes) {
            if (isOpenToOutside(graph, node)) {
                continue;
            }
            double dist = distFromTC.getOrDefault(node.id, Double.MAX_VALUE);
            if (isFinitePositive(dist) && dist < best) {
                best = dist;
            }
        }
        return best;
    }

    private boolean isFinitePositive(double value) {
        return value > 0.0 && value < Double.MAX_VALUE / 2;
    }

    /**
     * Returns true when outside can reach the target through free raid edges.
     */
    public static boolean isOpenToOutside(HouseGraph graph, TileNode target) {
        if (target == null) {
            return true;
        }

        Set<NodeKey> visited = new HashSet<>();
        ArrayDeque<NodeKey> queue = new ArrayDeque<>();

        TileNode outside = graph.getOutsideNode();
        queue.add(outside.id);
        visited.add(outside.id);

        Map<NodeKey, TileNode> nodeMap = new HashMap<>();
        for (TileNode n : graph.getAllNodes()) {
            nodeMap.put(n.id, n);
        }

        while (!queue.isEmpty()) {
            NodeKey curId = queue.poll();
            if (curId.equals(target.id)) {
                return true;
            }

            TileNode curNode = nodeMap.get(curId);
            if (curNode == null) {
                continue;
            }

            for (TileEdge edge : graph.getEdges(curNode)) {
                if (edge.raidSulfurCost == 0 && visited.add(edge.to.id)) {
                    queue.add(edge.to.id);
                }
            }
        }
        return false;
    }
}
