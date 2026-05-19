package com.rustbuilder.ai.rl.supervisor.application;

import com.rustbuilder.ai.core.TrainingMetrics;

/**
 * Compares a protected baseline branch against an LLM-tuned candidate branch.
 */
public class RLBranchComparator {
    private static final double MIN_BEST_SCORE_DELTA = 0.02;
    private static final double MAX_INVALID_RATE_REGRESSION = 0.05;

    public BranchComparison compare(TrainingMetrics baseline, TrainingMetrics candidate) {
        if (baseline == null || candidate == null) {
            return BranchComparison.keepBaseline("Missing branch metrics.");
        }

        double bestScoreDelta = candidate.bestScore - baseline.bestScore;
        double avgEvalDelta = candidate.avgEvalScore - baseline.avgEvalScore;
        double invalidRateDelta = candidate.invalidActionRate - baseline.invalidActionRate;

        boolean candidateHasUsefulBase = candidate.bestBaseBlocks >= baseline.bestBaseBlocks
            && (!baseline.bestBaseHasTC || candidate.bestBaseHasTC);
        boolean scoreImproved = bestScoreDelta >= MIN_BEST_SCORE_DELTA || avgEvalDelta > 0.0;
        boolean invalidRateAcceptable = invalidRateDelta <= MAX_INVALID_RATE_REGRESSION;

        if (candidateHasUsefulBase && scoreImproved && invalidRateAcceptable) {
            return BranchComparison.promoteCandidate(bestScoreDelta, avgEvalDelta, invalidRateDelta,
                "Candidate improved score without unacceptable invalid-action regression.");
        }

        return BranchComparison.keepBaseline(bestScoreDelta, avgEvalDelta, invalidRateDelta,
            "Candidate has not cleared promotion thresholds.");
    }

    public static final class BranchComparison {
        public final boolean promoteCandidate;
        public final double bestScoreDelta;
        public final double avgEvalDelta;
        public final double invalidRateDelta;
        public final String reason;

        private BranchComparison(boolean promoteCandidate,
                                 double bestScoreDelta,
                                 double avgEvalDelta,
                                 double invalidRateDelta,
                                 String reason) {
            this.promoteCandidate = promoteCandidate;
            this.bestScoreDelta = bestScoreDelta;
            this.avgEvalDelta = avgEvalDelta;
            this.invalidRateDelta = invalidRateDelta;
            this.reason = reason != null ? reason : "";
        }

        public static BranchComparison promoteCandidate(double bestScoreDelta,
                                                        double avgEvalDelta,
                                                        double invalidRateDelta,
                                                        String reason) {
            return new BranchComparison(true, bestScoreDelta, avgEvalDelta, invalidRateDelta, reason);
        }

        public static BranchComparison keepBaseline(String reason) {
            return new BranchComparison(false, 0.0, 0.0, 0.0, reason);
        }

        public static BranchComparison keepBaseline(double bestScoreDelta,
                                                    double avgEvalDelta,
                                                    double invalidRateDelta,
                                                    String reason) {
            return new BranchComparison(false, bestScoreDelta, avgEvalDelta, invalidRateDelta, reason);
        }
    }
}
