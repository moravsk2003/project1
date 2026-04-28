package com.rustbuilder.service.evaluator;

import com.rustbuilder.service.graph.*;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import com.rustbuilder.service.graph.HouseGraph.TileEdge;
import com.rustbuilder.service.graph.HouseGraph.TileNode;

/**
 * Evaluates logistics: shortest walk paths between key locations in 3D.
 * Uses Dijkstra on the walk-cost graph (doors=passable, walls=blocked, stairs=connect floors).
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
            return String.format("Logistics: Entrance→TC=%.1f, Entrance→WB=%.1f, Entrance→LR=%.1f, TC→LR=%.1f | Score=%.2f",
                    entranceToTC, entranceToWorkbench, entranceToLootRoom, tcToLootRoom, score);
        }
    }

    /**
     * Evaluate logistics for the given house graph.
     * The "entrance" is the outside node — the shortest walkable path from outside.
     */
    public LogisticsResult evaluate(HouseGraph graph) {
        TileNode outside = graph.getOutsideNode();
        TileNode tc = graph.findNodeByType("tc");
        TileNode workbench = graph.findNodeByType("workbench");
        TileNode lootRoom = graph.findNodeByType("loot_room");

        // No TC placed — logistics is impossible, score = 0
        if (tc == null) {
            return new LogisticsResult(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE, 0.0);
        }

        // Dijkstra from outside (entrance)
        Map<NodeKey, Double> distFromOutside = dijkstraWalk(graph, outside);

        double dTC = distFromOutside.getOrDefault(tc.id, Double.MAX_VALUE);
        double dWB = workbench != null ? distFromOutside.getOrDefault(workbench.id, Double.MAX_VALUE) : Double.MAX_VALUE;
        double dLR = lootRoom != null ? distFromOutside.getOrDefault(lootRoom.id, Double.MAX_VALUE) : Double.MAX_VALUE;

        // Dijkstra from TC to loot room
        double dTCtoLR = Double.MAX_VALUE;
        if (lootRoom != null) {
            Map<NodeKey, Double> distFromTC = dijkstraWalk(graph, tc);
            dTCtoLR = distFromTC.getOrDefault(lootRoom.id, Double.MAX_VALUE);
        }

        // Calculate score: inverse of total distance, normalized
        double totalDist = 0;
        int pathCount = 0;
        if (dTC < Double.MAX_VALUE) { totalDist += dTC; pathCount++; }
        if (dWB < Double.MAX_VALUE) { totalDist += dWB; pathCount++; }
        if (dLR < Double.MAX_VALUE) { totalDist += dLR; pathCount++; }
        if (dTCtoLR < Double.MAX_VALUE) { totalDist += dTCtoLR; pathCount++; }

        double score;
        if (pathCount == 0) {
            score = 0.0; // No valid paths
        } else {
            double avgDist = totalDist / pathCount;
            // Normalize: score = 1 / (1 + avgDist/10). Less penalty for distance so doors aren't punished.
            score = 1.0 / (1.0 + avgDist / 10.0);
            
            // Heuristic penalty: If the base is wide open (TC reachable without destroying anything)
            // It means it reached the TC without going through any doors/walls.
            if (isOpenToOutside(graph, tc)) {
                score *= 0.1;
            }
        }

        return new LogisticsResult(dTC, dWB, dLR, dTCtoLR, score);
    }

    private static class DistNode implements Comparable<DistNode> {
        final NodeKey id;
        final double dist;
        DistNode(NodeKey id, double dist) { this.id = id; this.dist = dist; }
        @Override public int compareTo(DistNode o) { return Double.compare(this.dist, o.dist); }
    }

    /**
     * Dijkstra shortest path using walk costs (only passable edges).
     */
    private Map<NodeKey, Double> dijkstraWalk(HouseGraph graph, TileNode start) {
        Map<NodeKey, Double> dist = new HashMap<>();
        dist.put(start.id, 0.0);

        Map<NodeKey, TileNode> nodeMap = new HashMap<>();
        for (TileNode n : graph.getAllNodes()) {
            nodeMap.put(n.id, n);
        }

        Set<NodeKey> visited = new HashSet<>();

        PriorityQueue<DistNode> dpq = new PriorityQueue<>();
        dpq.add(new DistNode(start.id, 0.0));

        while (!dpq.isEmpty()) {
            DistNode current = dpq.poll();
            double curDist = current.dist;
            NodeKey curId = current.id;

            if (visited.contains(curId)) continue;
            visited.add(curId);

            TileNode curNode = nodeMap.get(curId);
            if (curNode == null) continue;

            for (TileEdge edge : graph.getEdges(curNode)) {
                if (edge.walkCost >= Double.MAX_VALUE / 2) continue;

                double newDist = curDist + edge.walkCost;
                NodeKey neighborId = edge.to.id;

                if (newDist < dist.getOrDefault(neighborId, Double.MAX_VALUE)) {
                    dist.put(neighborId, newDist);
                    dpq.add(new DistNode(neighborId, newDist));
                }
            }
        }

        return dist;
    }

    /**
     * Checks if a target node is reachable from the outside without passing through any
     * structure that costs sulfur (like walls or locked doors).
     */
    private boolean isOpenToOutside(HouseGraph graph, TileNode target) {
        if (target == null) return true;
        Set<NodeKey> visited = new HashSet<>();
        java.util.Queue<NodeKey> queue = new java.util.LinkedList<>();
        
        TileNode outside = graph.getOutsideNode();
        queue.add(outside.id);
        visited.add(outside.id);
        
        Map<NodeKey, TileNode> nodeMap = new HashMap<>();
        for (TileNode n : graph.getAllNodes()) {
            nodeMap.put(n.id, n);
        }
        
        while(!queue.isEmpty()) {
            NodeKey curId = queue.poll();
            if (curId.equals(target.id)) return true;
            
            TileNode curNode = nodeMap.get(curId);
            if (curNode == null) continue;
            
            for (TileEdge edge : graph.getEdges(curNode)) {
                // Determine if we can freely pass this edge
                // An edge is free if raidSulfurCost == 0
                if (edge.raidSulfurCost == 0 && !visited.contains(edge.to.id)) {
                    visited.add(edge.to.id);
                    queue.add(edge.to.id);
                }
            }
        }
        return false;
    }
}
