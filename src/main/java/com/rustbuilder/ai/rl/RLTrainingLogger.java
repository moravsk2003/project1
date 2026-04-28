package com.rustbuilder.ai.rl;

import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.ai.rl.multidiscrete.HeuristicMaskingUtils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Locale;

/**
 * Handles all CSV file logging for {@link RLTrainingService}.
 *
 * <p>Manages three log files:
 * <ul>
 *   <li><b>epoch log</b> — one row per epoch with aggregated stats</li>
 *   <li><b>episode log</b> — one row per episode with detailed step/reward breakdown</li>
 *   <li><b>invalid action log</b> — one row per invalid placement attempt (for debugging)</li>
 * </ul>
 *
 * <p>Writers are opened once at the start of a training run ({@link #init()}) and
 * closed in a {@code finally} block ({@link #close()}). This avoids repeatedly
 * opening/closing file handles in the hot training loop.
 *
 * <p>The CSV format (column order, delimiter, header) is <em>identical</em> to the
 * format previously embedded in {@code RLTrainingService} — no behavioural change.
 */
public class RLTrainingLogger {

    private static final DateTimeFormatter TS_FMT =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private Path logFilePath;
    private Path invalidLogFilePath;
    private Path episodeLogFilePath;

    private PrintWriter epochLogWriter;
    private PrintWriter invalidLogWriter;
    private PrintWriter episodeLogWriter;

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    /**
     * Configures log file paths and writes CSV headers if the files do not yet exist.
     * Must be called before {@link #init()}.
     *
     * @param modelName     base name for the log files (e.g. the model identifier)
     * @param multiDiscrete {@code true} selects the multi-discrete suffix, {@code false} the legacy suffix
     */
    public void setLogFile(String modelName, boolean multiDiscrete) {
        Path dir = Paths.get("models_rl");
        try { Files.createDirectories(dir); } catch (IOException ignored) {}

        String suffix = multiDiscrete ? "_multi_discrete_training.csv" : "_legacy_training.csv";
        logFilePath = dir.resolve(modelName + suffix);

        if (!Files.exists(logFilePath)) {
            try (PrintWriter pw = new PrintWriter(new FileWriter(logFilePath.toFile()))) {
                pw.println("timestamp,epoch,episodes,avg_reward,best_reward_all_time,best_ep_reward,epsilon,loss,"
                         + "invalid_rate_pct,total_blocks,ram_mb,best_base_blocks,best_base_tc,best_base_doors,epoch_time_sec");
            } catch (IOException ignored) {}
        }

        invalidLogFilePath = dir.resolve(modelName + "_invalid_actions.csv");
        if (!Files.exists(invalidLogFilePath)) {
            try (PrintWriter pw = new PrintWriter(new FileWriter(invalidLogFilePath.toFile()))) {
                pw.println("timestamp,episode,step,fail_reason,action_type,floor,tileX,tileY,rotation,aimSector,minDist,socketDist");
            } catch (IOException ignored) {}
        }

        episodeLogFilePath = dir.resolve(modelName + "_episodes.csv");
        if (!Files.exists(episodeLogFilePath)) {
            try (PrintWriter pw = new PrintWriter(new FileWriter(episodeLogFilePath.toFile()))) {
                pw.println("timestamp,epoch,episode,total_episode_count,total_actions,invalid_actions,invalid_pct,"
                         + "blocks_placed,acc_step_reward,final_eval_reward,total_reward,epsilon,max_floor,"
                         + "floor_distribution,pruned_types,dead_tiles,dead_rotations,rejected_rot_aim,rejected_proximity");
            } catch (IOException ignored) {}
        }
    }

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    /**
     * Opens (or re-opens) all configured log writers in append mode.
     * Call once at the start of a training run.
     */
    public void init() {
        try {
            if (logFilePath != null) {
                epochLogWriter = new PrintWriter(new BufferedWriter(new FileWriter(logFilePath.toFile(), true)));
            }
            if (invalidLogFilePath != null) {
                invalidLogWriter = new PrintWriter(new BufferedWriter(new FileWriter(invalidLogFilePath.toFile(), true)));
            }
            if (episodeLogFilePath != null) {
                episodeLogWriter = new PrintWriter(new BufferedWriter(new FileWriter(episodeLogFilePath.toFile(), true)));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Flushes and closes all open writers.
     * Call in a {@code finally} block at the end of a training run.
     */
    public void close() {
        if (epochLogWriter != null) {
            try { epochLogWriter.close(); } catch (Exception ignored) {}
            epochLogWriter = null;
        }
        if (invalidLogWriter != null) {
            try { invalidLogWriter.close(); } catch (Exception ignored) {}
            invalidLogWriter = null;
        }
        if (episodeLogWriter != null) {
            try { episodeLogWriter.close(); } catch (Exception ignored) {}
            episodeLogWriter = null;
        }
    }

    // -------------------------------------------------------------------------
    // Write methods
    // -------------------------------------------------------------------------

    public void writeEpochLog(int epoch, int episodesInEpoch, double epochTotalReward,
                              int epochInvalid, int epochActions, int epochBlocks,
                              double epochBestEpScore, long epochMs,
                              double bestScore, double epsilon, double lastTrainLoss,
                              int bestBaseBlocks, boolean bestBaseHasTC, int bestBaseDoors) {
        if (epochLogWriter == null) return;

        double epochAvgReward = episodesInEpoch > 0 ? epochTotalReward / episodesInEpoch : 0;
        double invalidPct = epochActions > 0 ? (double) epochInvalid / epochActions * 100.0 : 0;
        long ramMB = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024);
        String ts = LocalDateTime.now().format(TS_FMT);

        epochLogWriter.printf(Locale.US,
            "%s,%d,%d,%.4f,%.4f,%.4f,%.5f,%.6f,%.1f,%d,%d,%d,%s,%d,%.1f%n",
            ts, epoch, episodesInEpoch, epochAvgReward, bestScore, epochBestEpScore,
            epsilon, lastTrainLoss, invalidPct, epochBlocks, ramMB,
            bestBaseBlocks, bestBaseHasTC ? "YES" : "NO", bestBaseDoors,
            epochMs / 1000.0);
        epochLogWriter.flush();
    }

    public void writeInvalidActionLog(int step, int episode, String failReason,
                                      String actionType, int floor, int tileX, int tileY,
                                      int rotation, int aimSector,
                                      double minDist, double socketDist) {
        if (invalidLogWriter == null) return;
        String ts = LocalDateTime.now().format(TS_FMT);
        invalidLogWriter.printf(Locale.US, "%s,%d,%d,%s,%s,%d,%d,%d,%d,%d,%.4f,%.4f%n",
            ts, episode, step, failReason, actionType, floor, tileX, tileY,
            rotation, aimSector, minDist, socketDist);
    }

    public void writeEpisodeLog(int epoch, int epochEpisode, int totalEpisode,
                                int totalActions, int invalidActions, int blocksPlaced,
                                double accStepReward, double finalEvalReward,
                                double epsilon, java.util.List<BuildingBlock> allBlocks) {
        if (episodeLogWriter == null) return;

        double invalidPct = totalActions > 0 ? (double) invalidActions / totalActions * 100.0 : 0;
        double totalReward = accStepReward + finalEvalReward;
        String ts = LocalDateTime.now().format(TS_FMT);

        // Floor diagnostics
        int maxFloor = 0;
        int[] floorCounts = new int[8];
        for (BuildingBlock b : allBlocks) {
            int z = b.getZ();
            if (z >= 0 && z < 8) {
                floorCounts[z]++;
                if (z > maxFloor) maxFloor = z;
            }
        }
        StringBuilder floorDist = new StringBuilder();
        for (int f = 0; f <= maxFloor; f++) {
            if (f > 0) floorDist.append("|");
            floorDist.append("F").append(f).append(":").append(floorCounts[f]);
        }
        if (floorDist.length() == 0) floorDist.append("empty");

        int prunedTypes    = HeuristicMaskingUtils.prunedTypeCount;
        int deadTiles      = HeuristicMaskingUtils.emptyTilesCount;
        int deadRots       = HeuristicMaskingUtils.emptyRotationsCount;
        int rejectedRotAim = HeuristicMaskingUtils.rotationsRejectedByAimCount;
        int rejectedProx   = HeuristicMaskingUtils.tilesRejectedByNearStructureRule;

        episodeLogWriter.printf(Locale.US,
            "%s,%d,%d,%d,%d,%d,%.1f,%d,%.4f,%.4f,%.4f,%.5f,%d,%s,%d,%d,%d,%d,%d%n",
            ts, epoch, epochEpisode, totalEpisode, totalActions, invalidActions, invalidPct,
            blocksPlaced, accStepReward, finalEvalReward, totalReward, epsilon,
            maxFloor, floorDist.toString(), prunedTypes, deadTiles, deadRots,
            rejectedRotAim, rejectedProx);
        episodeLogWriter.flush();

        if (invalidLogWriter != null) {
            invalidLogWriter.flush();
        }
    }
}
