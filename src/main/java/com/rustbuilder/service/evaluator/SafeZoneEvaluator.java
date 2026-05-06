package com.rustbuilder.service.evaluator;

import com.rustbuilder.service.graph.HouseGraph;
import com.rustbuilder.service.graph.HouseGraph.TileEdge;
import com.rustbuilder.service.graph.HouseGraph.TileNode;
import com.rustbuilder.service.graph.NodeKey;
import java.util.ArrayDeque;
import java.util.HashSet;
import java.util.Set;

/**
 * Rewards every closed tile that cannot be reached from outside through open passages.
 */
public class SafeZoneEvaluator {

    private static final double EPS = 1e-9;
    private static final double CLOSED_BLOCKS_FOR_FULL_SCORE = 10.0;

    public static class SafeZoneResult {
        public final int closedBlocks;
        public final double rawScore;
        public final double score;

        public SafeZoneResult(int closedBlocks, double rawScore, double score) {
            this.closedBlocks = closedBlocks;
            this.rawScore = rawScore;
            this.score = score;
        }

        @Override
        public String toString() {
            return String.format(
                "Safe Zone: closedBlocks=%d, raw=%.2f, score=%.3f",
                closedBlocks, rawScore, score
            );
        }
    }

    public SafeZoneResult evaluate(HouseGraph graph) {
        int closedBlocks = countClosedBlocks(graph);
        double raw = closedBlocks;
        double score = normalize(raw);
        return new SafeZoneResult(closedBlocks, raw, score);
    }

    private int countClosedBlocks(HouseGraph graph) {
        TileNode outside = graph.getOutsideNode();
        Set<TileNode> reachable = new HashSet<>();
        ArrayDeque<TileNode> queue = new ArrayDeque<>();
        queue.add(outside);
        reachable.add(outside);

        while (!queue.isEmpty()) {
            TileNode cur = queue.poll();
            Set<NodeKey> seenTargets = new HashSet<>();
            for (TileEdge edge : graph.getEdges(cur)) {
                if (!isOpenPassage(edge)) {
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

        int closedBlocks = 0;
        for (TileNode node : graph.getAllNodes()) {
            // HouseGraph marks roofless tiles as reachable from outside with an open edge.
            if ("tile".equals(node.type) && !reachable.contains(node)) {
                closedBlocks++;
            }
        }
        return closedBlocks;
    }

    private boolean isOpenPassage(TileEdge edge) {
        return edge.walkCost < Double.MAX_VALUE - EPS && edge.blocker == null;
    }

    private double normalize(double raw) {
        if (raw <= 0) {
            return 0.0;
        }
        return Math.min(1.0, raw / CLOSED_BLOCKS_FOR_FULL_SCORE);
    }
}
