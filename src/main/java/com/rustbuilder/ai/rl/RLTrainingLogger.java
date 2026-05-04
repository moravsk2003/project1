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
    private static final int EPISODE_LOG_FULL_DETAIL_UNTIL = 100;
    private static final int EPISODE_LOG_INTERVAL = 10;
    private static final int PERFORMANCE_LOG_FULL_DETAIL_UNTIL = 50;
    private static final int PERFORMANCE_LOG_INTERVAL = 25;
    private static final int INVALID_LOG_FULL_DETAIL_UNTIL = 50;
    private static final int INVALID_LOG_INTERVAL = 25;
    private static final int INVALID_LOG_MAX_ROWS_PER_EPISODE = 3;

    private Path logFilePath;
    private Path invalidLogFilePath;
    private Path episodeLogFilePath;
    private Path performanceLogFilePath;

    private PrintWriter epochLogWriter;
    private PrintWriter invalidLogWriter;
    private PrintWriter episodeLogWriter;
    private PrintWriter performanceLogWriter;

    private String currentRunId = "unknown";
    private String currentEncoderVersion = "0";
    private int currentVoxelChannels = 0;
    private boolean currentHasGlobal = false;
    private int currentGlobalCount = 0;
    private int lastInvalidLogEpisode = -1;
    private int invalidRowsForEpisode = 0;

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
                pw.println("timestamp,epoch,episodes,state_encoder_version,voxel_channels,has_global_vector,global_feature_count,"
                         + "avg_step_reward,avg_final_eval_reward,avg_total_reward,positive_total_rate,positive_final_rate,"
                         + "avg_blocks_placed,avg_rejected_proximity,avg_encoder_time_ms,avg_global_encoder_time_ms,"
                         + "best_reward_all_time,best_ep_reward,epsilon,loss,invalid_rate_pct,total_blocks,ram_mb,"
                         + "best_base_blocks,best_base_tc,best_base_doors,epoch_time_sec");
            } catch (IOException ignored) {}
        }

        invalidLogFilePath = dir.resolve(modelName + "_invalid_actions.csv");
        if (!Files.exists(invalidLogFilePath)) {
            try (PrintWriter pw = new PrintWriter(new FileWriter(invalidLogFilePath.toFile()))) {
                pw.println("timestamp,run_id,epoch,episode,step,state_encoder_version,fail_reason,action_type,floor,tileX,tileY,rotation,aimSector,minDist,socketDist,mask_allowed,physics_allowed");
            } catch (IOException ignored) {}
        }

        episodeLogFilePath = dir.resolve(modelName + "_episodes.csv");
        if (!Files.exists(episodeLogFilePath)) {
            try (PrintWriter pw = new PrintWriter(new FileWriter(episodeLogFilePath.toFile()))) {
                pw.println("timestamp,run_id,epoch,episode,total_episode_count,state_encoder_version,voxel_channels,has_global_vector,global_feature_count,"
                         + "total_actions,invalid_actions,invalid_pct,blocks_placed,acc_step_reward,final_eval_reward,total_reward,epsilon,"
                         + "stop_reason,max_floor,placed_floor_distribution,rejected_total,rejected_no_support,rejected_collision,"
                         + "rejected_bad_socket,rejected_out_of_bounds,pruned_types,dead_tiles,dead_rotations,rejected_rot_aim,rejected_proximity,"
                         + "encoder_time_ms,global_encoder_time_ms,global_has_nan,global_has_infinity,"
                         + "global_feat_cnt,global_feat_mean,global_feat_max,global_feat_nonzero");
            } catch (IOException ignored) {}
        }

        performanceLogFilePath = dir.resolve(modelName + "_performance_tmp.csv");
        if (!Files.exists(performanceLogFilePath)) {
            try (PrintWriter pw = new PrintWriter(new FileWriter(performanceLogFilePath.toFile()))) {
                pw.println("timestamp,run_id,epoch,episode,total_episode_count,state_encoder_version,total_actions,blocks_placed,invalid_actions,"
                         + "episode_ms,context_ms,state_encode_ms,next_state_encode_ms,action_select_ms,grid_clone_ms,placement_ms,finalize_ms,"
                         + "reward_ms,replay_ms,train_batch_ms,invalid_log_ms,final_eval_ms,episode_log_ms,update_best_ms,untracked_ms,"
                         + "top_stage,top_stage_ms,rejected_proximity,rejected_rot_aim,dead_tiles,dead_rotations,pruned_types");
            } catch (IOException ignored) {}
        }
    }

    public void setRunContext(String runId, String encoderVersion, int voxelChannels, boolean hasGlobal, int globalCount) {
        this.currentRunId = runId;
        this.currentEncoderVersion = encoderVersion;
        this.currentVoxelChannels = voxelChannels;
        this.currentHasGlobal = hasGlobal;
        this.currentGlobalCount = globalCount;
    }

    public void writeRunMetadata(String modelName, com.rustbuilder.ai.rl.env.spec.EncodingRuntimeConfig config,
                                 String rewardConfigName, String trainingConfigName,
                                 String logsDir, String modelsDir) {
        Path dir = Paths.get("models_rl");
        Path metaPath = dir.resolve(modelName + "_run_metadata.json");

        try (PrintWriter pw = new PrintWriter(new FileWriter(metaPath.toFile()))) {
            pw.println("{");
            pw.printf(Locale.US, "  \"run_id\": \"%s\",%n", currentRunId);
            pw.printf(Locale.US, "  \"model_name\": \"%s\",%n", modelName);
            pw.printf(Locale.US, "  \"timestamp_start\": \"%s\",%n", LocalDateTime.now().format(TS_FMT));
            pw.println();
            pw.printf(Locale.US, "  \"state_encoder_name\": \"%s\",%n", config.stateEncodingSpec.encoderName);
            pw.printf(Locale.US, "  \"state_encoder_version\": \"%s\",%n", currentEncoderVersion);
            pw.printf(Locale.US, "  \"requested_state_encoder_version\": \"%s\",%n", config.stateEncodingSpec.encoderVersion);
            pw.printf(Locale.US, "  \"actual_state_encoder_version\": \"%s\",%n", currentEncoderVersion);
            pw.printf(Locale.US, "  \"voxel_channels\": %d,%n", currentVoxelChannels);
            pw.println();
            pw.printf(Locale.US, "  \"has_global_vector\": %b,%n", currentHasGlobal);
            pw.printf(Locale.US, "  \"global_feature_count\": %d,%n", currentGlobalCount);
            pw.println();
            pw.printf(Locale.US, "  \"grid_width\": %d,%n", config.gridSpec.width);
            pw.printf(Locale.US, "  \"grid_height\": %d,%n", config.gridSpec.height);
            pw.printf(Locale.US, "  \"grid_floors\": %d,%n", config.gridSpec.floors);
            pw.println();
            pw.printf(Locale.US, "  \"action_type_count\": %d,%n", config.actionSpaceSpec.typeCount);
            pw.printf(Locale.US, "  \"action_floor_count\": %d,%n", config.actionSpaceSpec.floorCount);
            pw.printf(Locale.US, "  \"action_tile_count\": %d,%n", config.actionSpaceSpec.tileCount);
            pw.printf(Locale.US, "  \"action_rotation_count\": %d,%n", config.actionSpaceSpec.rotationCount);
            pw.printf(Locale.US, "  \"action_aim_count\": %d,%n", config.actionSpaceSpec.aimCount);
            pw.printf(Locale.US, "  \"tile_indexing_mode\": \"LEGACY_X_MAJOR\",%n"); // Hardcoded for now as it's the only one
            pw.println();
            pw.printf(Locale.US, "  \"reward_config_name\": \"%s\",%n", rewardConfigName);
            pw.printf(Locale.US, "  \"training_config_name\": \"%s\",%n", trainingConfigName);
            pw.printf(Locale.US, "  \"episode_log_policy\": \"first_%d_then_every_%d\",%n",
                EPISODE_LOG_FULL_DETAIL_UNTIL, EPISODE_LOG_INTERVAL);
            pw.printf(Locale.US, "  \"performance_log_policy\": \"first_%d_then_every_%d\",%n",
                PERFORMANCE_LOG_FULL_DETAIL_UNTIL, PERFORMANCE_LOG_INTERVAL);
            pw.printf(Locale.US, "  \"invalid_action_log_policy\": \"first_%d_then_every_%d_max_%d_rows_per_episode\",%n",
                INVALID_LOG_FULL_DETAIL_UNTIL, INVALID_LOG_INTERVAL, INVALID_LOG_MAX_ROWS_PER_EPISODE);
            pw.printf(Locale.US, "  \"logs_output_dir\": \"%s\",%n", logsDir);
            pw.printf(Locale.US, "  \"model_output_dir\": \"%s\"%n", modelsDir);
            pw.println("}");
        } catch (IOException e) {
            e.printStackTrace();
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
            if (performanceLogFilePath != null) {
                performanceLogWriter = new PrintWriter(new BufferedWriter(new FileWriter(performanceLogFilePath.toFile(), true)));
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
        if (performanceLogWriter != null) {
            try { performanceLogWriter.close(); } catch (Exception ignored) {}
            performanceLogWriter = null;
        }
    }

    // -------------------------------------------------------------------------
    // Write methods
    // -------------------------------------------------------------------------

    public void writeEpochLog(int epoch, int episodesInEpoch, double epochTotalReward,
                              double epochTotalStepReward, double epochTotalFinalReward,
                              int epochPositiveTotal, int epochPositiveFinal,
                              int epochInvalid, int epochActions, int epochBlocks,
                              int epochRejectedProx, long epochEncoderTime, long epochGlobalTime,
                              double epochBestEpScore, long epochMs,
                              double bestScore, double epsilon, double lastTrainLoss,
                              int bestBaseBlocks, boolean bestBaseHasTC, int bestBaseDoors) {
        if (epochLogWriter == null) return;

        double avgTotalReward = episodesInEpoch > 0 ? epochTotalReward / episodesInEpoch : 0;
        double avgStepReward = episodesInEpoch > 0 ? epochTotalStepReward / episodesInEpoch : 0;
        double avgFinalReward = episodesInEpoch > 0 ? epochTotalFinalReward / episodesInEpoch : 0;

        double positiveTotalRate = episodesInEpoch > 0 ? (double) epochPositiveTotal / episodesInEpoch : 0;
        double positiveFinalRate = episodesInEpoch > 0 ? (double) epochPositiveFinal / episodesInEpoch : 0;

        double avgBlocks = episodesInEpoch > 0 ? (double) epochBlocks / episodesInEpoch : 0;
        double avgRejectedProx = episodesInEpoch > 0 ? (double) epochRejectedProx / episodesInEpoch : 0;
        double avgEncoderTime = episodesInEpoch > 0 ? (double) epochEncoderTime / episodesInEpoch : 0;
        double avgGlobalTime = episodesInEpoch > 0 ? (double) epochGlobalTime / episodesInEpoch : 0;

        double invalidPct = epochActions > 0 ? (double) epochInvalid / epochActions * 100.0 : 0;
        long ramMB = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024);
        String ts = LocalDateTime.now().format(TS_FMT);

        epochLogWriter.printf(Locale.US,
            "%s,%d,%d,%s,%d,%b,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.5f,%.6f,%.1f,%d,%d,%d,%s,%d,%.1f%n",
            ts, epoch, episodesInEpoch, currentEncoderVersion, currentVoxelChannels, currentHasGlobal, currentGlobalCount,
            avgStepReward, avgFinalReward, avgTotalReward, positiveTotalRate, positiveFinalRate,
            avgBlocks, avgRejectedProx, avgEncoderTime, avgGlobalTime,
            bestScore, epochBestEpScore, epsilon, lastTrainLoss, invalidPct, epochBlocks, ramMB,
            bestBaseBlocks, bestBaseHasTC ? "YES" : "NO", bestBaseDoors,
            epochMs / 1000.0);
        epochLogWriter.flush();
    }

    public void writeInvalidActionLog(int epoch, int episode, int step, String failReason,
                                      String actionType, int floor, int tileX, int tileY,
                                      int rotation, int aimSector,
                                      double minDist, double socketDist,
                                      boolean maskAllowed, boolean physicsAllowed) {
        if (!shouldWriteInvalidActionLog(episode, step)) return;
        String ts = LocalDateTime.now().format(TS_FMT);
        invalidLogWriter.printf(Locale.US, "%s,%s,%d,%d,%d,%s,%s,%s,%d,%d,%d,%d,%d,%.4f,%.4f,%b,%b%n",
            ts, currentRunId, epoch, episode, step, currentEncoderVersion, failReason, actionType, floor, tileX, tileY,
            rotation, aimSector, minDist, socketDist, maskAllowed, physicsAllowed);
        invalidRowsForEpisode++;
    }

    public void writeEpisodeLog(int epoch, int epochEpisode, int totalEpisode,
                                EpisodeResult result, double epsilon) {
        if (episodeLogWriter == null || !shouldWriteEpisodeLog(totalEpisode)) return;

        double invalidPct = result.totalActions > 0 ? (double) result.invalidActions / result.totalActions * 100.0 : 0;
        double totalReward = result.accStepReward + result.finalEvalReward;
        String ts = LocalDateTime.now().format(TS_FMT);

        // Placed floor distribution
        int maxFloor = 0;
        int[] floorCounts = new int[8];
        for (BuildingBlock b : result.grid.getAllBlocks()) {
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

        // Map error stats to columns
        int rejectedTotal     = result.invalidActions + rejectedProx + rejectedRotAim + prunedTypes + deadTiles + deadRots;
        int rejectedNoSupport = result.errorStats.getOrDefault(PlacementError.NO_SUPPORT, 0);
        int rejectedCollision = result.errorStats.getOrDefault(PlacementError.COLLISION, 0);
        int rejectedBadSocket = countBadSocketErrors(result);
        int rejectedOutOfBounds = result.errorStats.getOrDefault(PlacementError.OUT_OF_BOUNDS, 0);

        com.rustbuilder.ai.rl.env.state.GlobalFeatureDiagnostics gDiag = result.globalDiag;
        int gCnt = gDiag != null ? gDiag.globalFeatureCount : 0;
        double gMean = gDiag != null ? gDiag.globalMean : 0.0;
        double gMax = gDiag != null ? gDiag.globalMax : 0.0;
        int gNonzero = gDiag != null ? gDiag.globalNonzeroCount : 0;
        boolean gNaN = gDiag != null && gDiag.hasNaN;
        boolean gInf = gDiag != null && gDiag.hasInfinity;

        episodeLogWriter.printf(Locale.US,
            "%s,%s,%d,%d,%d,%s,%d,%b,%d,%d,%d,%.1f,%d,%.4f,%.4f,%.4f,%.5f,%s,%d,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%b,%b,%d,%.4f,%.4f,%d%n",
            ts, currentRunId, epoch, epochEpisode, totalEpisode,
            currentEncoderVersion, currentVoxelChannels, currentHasGlobal, currentGlobalCount,
            result.totalActions, result.invalidActions, invalidPct,
            result.blocksPlaced, result.accStepReward, result.finalEvalReward, totalReward, epsilon,
            result.stopReason.name(), maxFloor, floorDist.toString(),
            rejectedTotal, rejectedNoSupport, rejectedCollision, rejectedBadSocket, rejectedOutOfBounds,
            prunedTypes, deadTiles, deadRots, rejectedRotAim, rejectedProx,
            result.totalEncoderTimeMs, result.totalGlobalEncoderTimeMs, gNaN, gInf,
            gCnt, gMean, gMax, gNonzero);
        episodeLogWriter.flush();

        if (invalidLogWriter != null) {
            invalidLogWriter.flush();
        }
    }

    private int countBadSocketErrors(EpisodeResult result) {
        int total = 0;
        total += result.errorStats.getOrDefault(PlacementError.BAD_SOCKET, 0);
        total += result.errorStats.getOrDefault(PlacementError.BAD_SOCKET_IS_FIRST, 0);
        total += result.errorStats.getOrDefault(PlacementError.BAD_SOCKET_NO_TARGET, 0);
        total += result.errorStats.getOrDefault(PlacementError.BAD_SOCKET_WRONG_TARGET_TYPE, 0);
        total += result.errorStats.getOrDefault(PlacementError.BAD_SOCKET_NO_SOCKET_ALIGNMENT, 0);
        total += result.errorStats.getOrDefault(PlacementError.BAD_SOCKET_CENTERDIST_REJECT, 0);
        return total;
    }

    public void writePerformanceLog(int epoch, int epochEpisode, int totalEpisode,
                                    EpisodeResult result) {
        if (performanceLogWriter == null || result == null || !shouldWritePerformanceLog(totalEpisode)) return;

        long totalEpisodeNs = result.perfEpisodeNs
            + result.perfFinalEvalNs
            + result.perfEpisodeLogNs
            + result.perfUpdateBestNs;
        long trackedNs = result.perfContextNs
            + result.perfStateEncodeNs
            + result.perfNextStateEncodeNs
            + result.perfActionSelectNs
            + result.perfGridCloneNs
            + result.perfPlacementNs
            + result.perfFinalizeNs
            + result.perfRewardNs
            + result.perfReplayNs
            + result.perfTrainNs
            + result.perfInvalidLogNs
            + result.perfFinalEvalNs
            + result.perfEpisodeLogNs
            + result.perfUpdateBestNs;
        long untrackedNs = Math.max(0, totalEpisodeNs - trackedNs);

        StageMax top = maxStage(
            stage("context", result.perfContextNs),
            stage("state_encode", result.perfStateEncodeNs),
            stage("next_state_encode", result.perfNextStateEncodeNs),
            stage("action_select", result.perfActionSelectNs),
            stage("grid_clone", result.perfGridCloneNs),
            stage("placement", result.perfPlacementNs),
            stage("finalize", result.perfFinalizeNs),
            stage("reward", result.perfRewardNs),
            stage("replay", result.perfReplayNs),
            stage("train_batch", result.perfTrainNs),
            stage("invalid_log", result.perfInvalidLogNs),
            stage("final_eval", result.perfFinalEvalNs),
            stage("episode_log", result.perfEpisodeLogNs),
            stage("update_best", result.perfUpdateBestNs),
            stage("untracked", untrackedNs)
        );

        String ts = LocalDateTime.now().format(TS_FMT);
        performanceLogWriter.printf(Locale.US,
            "%s,%s,%d,%d,%d,%s,%d,%d,%d,"
            + "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
            + "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
            + "%s,%.3f,%d,%d,%d,%d,%d%n",
            ts, currentRunId, epoch, epochEpisode, totalEpisode, currentEncoderVersion,
            result.totalActions, result.blocksPlaced, result.invalidActions,
            nsToMs(totalEpisodeNs),
            nsToMs(result.perfContextNs),
            nsToMs(result.perfStateEncodeNs),
            nsToMs(result.perfNextStateEncodeNs),
            nsToMs(result.perfActionSelectNs),
            nsToMs(result.perfGridCloneNs),
            nsToMs(result.perfPlacementNs),
            nsToMs(result.perfFinalizeNs),
            nsToMs(result.perfRewardNs),
            nsToMs(result.perfReplayNs),
            nsToMs(result.perfTrainNs),
            nsToMs(result.perfInvalidLogNs),
            nsToMs(result.perfFinalEvalNs),
            nsToMs(result.perfEpisodeLogNs),
            nsToMs(result.perfUpdateBestNs),
            nsToMs(untrackedNs),
            top.name,
            nsToMs(top.ns),
            HeuristicMaskingUtils.tilesRejectedByNearStructureRule,
            HeuristicMaskingUtils.rotationsRejectedByAimCount,
            HeuristicMaskingUtils.emptyTilesCount,
            HeuristicMaskingUtils.emptyRotationsCount,
            HeuristicMaskingUtils.prunedTypeCount);
        performanceLogWriter.flush();
    }

    private static double nsToMs(long ns) {
        return ns / 1_000_000.0;
    }

    private static StageMax stage(String name, long ns) {
        return new StageMax(name, ns);
    }

    private static StageMax maxStage(StageMax... stages) {
        StageMax best = stages[0];
        for (int i = 1; i < stages.length; i++) {
            if (stages[i].ns > best.ns) {
                best = stages[i];
            }
        }
        return best;
    }

    public boolean shouldWriteInvalidActionLog(int episode, int step) {
        if (invalidLogWriter == null) return false;
        if (!shouldSampleEpisode(episode, INVALID_LOG_FULL_DETAIL_UNTIL, INVALID_LOG_INTERVAL)) return false;

        if (episode != lastInvalidLogEpisode) {
            lastInvalidLogEpisode = episode;
            invalidRowsForEpisode = 0;
        }

        return invalidRowsForEpisode < INVALID_LOG_MAX_ROWS_PER_EPISODE;
    }

    private static boolean shouldWriteEpisodeLog(int totalEpisode) {
        return shouldSampleEpisode(totalEpisode, EPISODE_LOG_FULL_DETAIL_UNTIL, EPISODE_LOG_INTERVAL);
    }

    private static boolean shouldWritePerformanceLog(int totalEpisode) {
        return shouldSampleEpisode(totalEpisode, PERFORMANCE_LOG_FULL_DETAIL_UNTIL, PERFORMANCE_LOG_INTERVAL);
    }

    private static boolean shouldSampleEpisode(int episode, int fullDetailUntil, int interval) {
        return episode <= fullDetailUntil || (interval > 0 && episode % interval == 0);
    }

    private static class StageMax {
        final String name;
        final long ns;

        StageMax(String name, long ns) {
            this.name = name;
            this.ns = ns;
        }
    }
}
