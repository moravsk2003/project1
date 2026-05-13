package com.rustbuilder.ai.rl.supervisor;

import com.rustbuilder.ai.core.TrainingMetrics;
import com.rustbuilder.ai.rl.RLModelManager;
import com.rustbuilder.ai.rl.RLRewardConfig;
import com.rustbuilder.ai.rl.RLTrainingConfig;
import com.rustbuilder.ai.rl.RLTrainingService;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Background orchestrator that autonomously controls the RLTrainingService.
 * When enabled, it periodically polls the LLM when idle to decide whether to start a new run.
 */
public class LlmOrchestrator {

    private final RLTrainingService rlService;
    private final SupervisorDecisionValidator decisionValidator = new SupervisorDecisionValidator();
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final AtomicBoolean idleDecisionInProgress = new AtomicBoolean(false);
    private Thread orchestratorThread;
    private volatile java.util.function.Consumer<String> statusCallback;
    private volatile java.util.function.Consumer<TrainingMetrics> trainingProgressCallback;
    
    // Interval to poll the LLM when idle (in milliseconds)
    private static final long IDLE_POLL_INTERVAL_MS = 10000;

    public LlmOrchestrator(RLTrainingService rlService) {
        this.rlService = rlService;
    }

    public void start() {
        if (running.compareAndSet(false, true)) {
            orchestratorThread = new Thread(this::runLoop);
            orchestratorThread.setDaemon(true);
            orchestratorThread.start();
        }
    }

    public void stop() {
        running.set(false);
        if (orchestratorThread != null) {
            orchestratorThread.interrupt();
        }
    }

    public void forceCheck(java.util.function.Consumer<String> statusCallback) {
        forceCheck(statusCallback, null);
    }

    public void forceCheck(java.util.function.Consumer<String> statusCallback,
                           java.util.function.Consumer<TrainingMetrics> trainingProgressCallback) {
        java.util.function.Consumer<String> cb = statusCallback != null ? statusCallback : s -> {};
        this.statusCallback = cb;
        this.trainingProgressCallback = trainingProgressCallback;
        if (!idleDecisionInProgress.compareAndSet(false, true)) {
            cb.accept("Orchestrator: LLM check already running.");
            return;
        }
        cb.accept("Orchestrator: contacting LLM...");
        Thread t = new Thread(() -> {
            try {
                LlmSupervisorConfig config = rlService.getSupervisorConfig();
                if (config == null || !config.isEnabled()) {
                    cb.accept("Orchestrator: supervisor is not enabled.");
                    return;
                }
                if (rlService.isTrainingRunning()) {
                    cb.accept("Orchestrator: training is already running.");
                    return;
                }
                SupervisorObservation obs = rlService.createIdleSupervisorObservation("idle_check");
                SupervisorDecision decision = askLlm(config, obs);
                processDecision(decision, obs, config, cb, 0);
            } catch (Exception e) {
                cb.accept("[ORCHESTRATOR] Error: " + e.getMessage());
            } finally {
                idleDecisionInProgress.set(false);
            }
        });
        t.setDaemon(true);
        t.start();
    }

    private void runLoop() {
        while (running.get()) {
            try {
                LlmSupervisorConfig config = rlService.getSupervisorConfig();
                if (config != null && config.isEnabled()) {
                    // If training is NOT currently running
                    if (!rlService.isTrainingRunning()) {
                        if (!idleDecisionInProgress.compareAndSet(false, true)) {
                            Thread.sleep(IDLE_POLL_INTERVAL_MS);
                            continue;
                        }
                        // Create a special observation indicating idle state
                        try {
                            SupervisorObservation obs = rlService.createIdleSupervisorObservation("idle_check");
                            
                            SupervisorDecision decision = askLlm(config, obs);
                            processDecision(decision, obs, config, msg -> {
                                // RunLoop doesn't have a UI callback, but we can log via the status callback if set
                                if (this.statusCallback != null) this.statusCallback.accept(msg);
                            }, 0);
                        } finally {
                            idleDecisionInProgress.set(false);
                        }
                    }
                }
                
                Thread.sleep(IDLE_POLL_INTERVAL_MS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                // Ignore transient errors and keep polling
                try {
                    Thread.sleep(IDLE_POLL_INTERVAL_MS);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }
    }

    private SupervisorDecision askLlm(LlmSupervisorConfig config, SupervisorObservation obs) {
        try {
            LlmSupervisor supervisor = LlmSupervisorFactory.create(
                config,
                () -> rlService.getRewardConfig(),
                createSupervisorEnvironmentOverrides(config));
            return decisionValidator.validate(supervisor.review(obs), config, rlService.getRewardConfig(), obs);
        } catch (Exception e) {
            rlService.setLastSupervisorDecision(null, "Orchestrator Error: " + e.getMessage());
            return null;
        }
    }

    private void processDecision(SupervisorDecision decision,
                                 SupervisorObservation originalObs,
                                 LlmSupervisorConfig config,
                                 java.util.function.Consumer<String> cb,
                                 int depth) {
        if (decision == null) {
            cb.accept("[ORCHESTRATOR] LLM returned no decision (check API key / command).");
            rlService.setLastSupervisorDecision(null, "Orchestrator: LLM returned no decision or error occurred.");
            return;
        }

        String msg = String.format("[ORCHESTRATOR] action=%s, reason=%s", decision.getAction().name(), decision.getReason());
        cb.accept(msg);
        rlService.setLastSupervisorDecision(decision, msg);
        if (decision.getProposedCallFrequency() != null && config.getApplyMode() == LlmSupervisorApplyMode.AUTO_APPLY) {
            boolean frequencyApplied = rlService.applySupervisorCallFrequency(decision.getProposedCallFrequency());
            if (frequencyApplied) {
                LlmSupervisorConfig updatedConfig = rlService.getSupervisorConfig();
                cb.accept(String.format(
                    "[ORCHESTRATOR] LLM call frequency set to %s; effective interval %d episodes.",
                    updatedConfig.getCallFrequency().name(),
                    updatedConfig.getEffectiveCallIntervalEpisodes()));
                config = updatedConfig;
            }
        }

        if (decision.getAction() == SupervisorAction.START_NEW_RUN || decision.getAction() == SupervisorAction.RESTART_TRAINING) {
            if (config.getApplyMode() != LlmSupervisorApplyMode.AUTO_APPLY) {
                cb.accept("[ORCHESTRATOR] Start-run decision recorded but not applied because apply mode is " + config.getApplyMode() + ".");
                return;
            }
            String modelName = decision.getProposedModelName();
            if (modelName == null || modelName.isBlank()) modelName = "auto_" + System.currentTimeMillis();
            cb.accept("[ORCHESTRATOR] Starting new run: " + modelName.trim());
            startNewRun(decision);
        } else if (decision.getAction() == SupervisorAction.PROMOTE_BRANCH) {
            if (config.getApplyMode() != LlmSupervisorApplyMode.AUTO_APPLY) {
                cb.accept("[ORCHESTRATOR] Promote-branch decision recorded but not applied because apply mode is " + config.getApplyMode() + ".");
                return;
            }
            boolean promoted = rlService.promoteLatestBranch();
            cb.accept(promoted
                ? "[ORCHESTRATOR] Candidate branch promoted."
                : "[ORCHESTRATOR] No candidate branch available to promote.");
        } else if (decision.getAction() == SupervisorAction.JUMP_TO_BRANCH) {
            if (config.getApplyMode() != LlmSupervisorApplyMode.AUTO_APPLY) {
                cb.accept("[ORCHESTRATOR] Branch-jump decision recorded but not applied because apply mode is " + config.getApplyMode() + ".");
                return;
            }
            boolean jumped = rlService.requestBranchJump(decision.getProposedModelName(), decision.getReason());
            cb.accept(jumped
                ? "[ORCHESTRATOR] Branch jump queued/applied: " + decision.getProposedModelName()
                : "[ORCHESTRATOR] Branch jump rejected: " + decision.getProposedModelName());
        } else if (decision.getAction() == SupervisorAction.REQUEST_HISTORICAL_REPORT) {
            if (depth >= 1) {
                cb.accept("[ORCHESTRATOR] Historical report already provided; ignoring repeated report request.");
                return;
            }
            cb.accept("[ORCHESTRATOR] LLM requested historical report for " + decision.getReportModelName());
            String report = generateHistoricalReport(decision.getReportModelName(), decision.getReportStartEpoch(), decision.getReportEndEpoch());
            
            SupervisorObservation newObs = new SupervisorObservation(
                originalObs.branchId, originalObs.totalEpisodesTrained, originalObs.bestScore, originalObs.avgEvalScore,
                originalObs.currentEpisodeEvalScore, originalObs.currentEpisodeStepReward, originalObs.currentEpisodeFinalReward,
                originalObs.bestTotalReward, originalObs.invalidActionRate, originalObs.lastEpisodeInvalidActions,
                originalObs.lastEpisodeTotalActions, originalObs.epsilon, originalObs.lastTrainLoss, originalObs.memorySize,
                originalObs.bestBaseBlocks, originalObs.bestBaseHasTC, originalObs.bestBaseDoors, originalObs.episodeBlocksPlaced,
                originalObs.episodeHasTC, originalObs.componentCount, originalObs.mainComponentBlocks, originalObs.evalLogisticsScore,
                originalObs.evalCostScore, originalObs.evalRaidScore, originalObs.evalWorkingAreaScore, originalObs.evalSafeZoneScore,
                originalObs.raidSulfurToTC, originalObs.stopReason, originalObs.stepRewardBreakdown, originalObs.finalRewardBreakdown,
                originalObs.trainingStartTimeIso, originalObs.currentTimeIso, originalObs.trainingDeadlineIso,
                originalObs.trainingElapsedMs, originalObs.trainingRemainingMs, originalObs.trainingTimeLimitEnabled,
                originalObs.trainingTimeLimitReached, originalObs.invalidActionReasons, originalObs.actionTypeCounts,
                originalObs.currentRewardConfig, originalObs.trainingContext, originalObs.trendMetrics, report
            );
            
            SupervisorDecision nextDecision = askLlm(config, newObs);
            processDecision(nextDecision, newObs, config, cb, depth + 1);
        }
    }

    private java.util.Map<String, String> createSupervisorEnvironmentOverrides(LlmSupervisorConfig config) {
        String apiKey = config != null ? config.getApiKey() : "";
        if (apiKey.isBlank()) {
            return java.util.Map.of();
        }
        java.util.Map<String, String> environment = new java.util.HashMap<>();
        environment.put("GEMINI_API_KEY", apiKey);
        environment.put("GOOGLE_API_KEY", apiKey);
        environment.put("LLM_API_KEY", apiKey);
        environment.put("OPENAI_API_KEY", apiKey);
        return environment;
    }

    private String generateHistoricalReport(String modelName, Integer startEpoch, Integer endEpoch) {
        if (modelName == null || modelName.isBlank()) return "Error: Model name not provided.";
        
        java.nio.file.Path logPath = RLModelManager.findExistingModelFile(modelName, modelName + "_legacy_training.csv");
        if (!java.nio.file.Files.exists(logPath)) {
            logPath = RLModelManager.findExistingModelFile(modelName, modelName + "_multi_discrete_training.csv");
            if (!java.nio.file.Files.exists(logPath)) {
                return "Error: Could not find training logs for model: " + modelName;
            }
        }
        
        try {
            java.util.List<String> lines = java.nio.file.Files.readAllLines(logPath);
            if (lines.size() <= 1) return "Error: Log file is empty.";
            
            String[] headers = lines.get(0).split(",");
            int epochIdx = -1, avgTotalRewardIdx = -1, bestEpRewardIdx = -1, epsilonIdx = -1, invalidRateIdx = -1;
            for (int i = 0; i < headers.length; i++) {
                if (headers[i].equals("epoch")) epochIdx = i;
                if (headers[i].equals("avg_total_reward")) avgTotalRewardIdx = i;
                if (headers[i].equals("best_ep_reward")) bestEpRewardIdx = i;
                if (headers[i].equals("epsilon")) epsilonIdx = i;
                if (headers[i].equals("invalid_rate_pct")) invalidRateIdx = i;
            }
            
            if (epochIdx == -1 || avgTotalRewardIdx == -1) return "Error: Invalid CSV format.";
            
            int sEpoch = startEpoch != null ? startEpoch : 0;
            int eEpoch = endEpoch != null ? endEpoch : Integer.MAX_VALUE;
            
            int count = 0;
            double startReward = 0, endReward = 0;
            double minReward = Double.MAX_VALUE, maxReward = -Double.MAX_VALUE;
            
            StringBuilder summary = new StringBuilder();
            summary.append("Historical Report for ").append(modelName).append(" (Epochs ").append(sEpoch).append("-").append(eEpoch == Integer.MAX_VALUE ? "End" : eEpoch).append("):\n");
            
            for (int i = 1; i < lines.size(); i++) {
                String[] parts = lines.get(i).split(",");
                if (parts.length <= avgTotalRewardIdx) continue;
                try {
                    int ep = Integer.parseInt(parts[epochIdx]);
                    if (ep >= sEpoch && ep <= eEpoch) {
                        double reward = Double.parseDouble(parts[avgTotalRewardIdx]);
                        if (count == 0) startReward = reward;
                        endReward = reward;
                        minReward = Math.min(minReward, reward);
                        maxReward = Math.max(maxReward, reward);
                        count++;
                        
                        // Sample every 10th epoch for trend
                        if (count <= 5 || count % 10 == 0 || i == lines.size() - 1) {
                            String eps = epsilonIdx != -1 ? parts[epsilonIdx] : "N/A";
                            String inv = invalidRateIdx != -1 ? parts[invalidRateIdx] : "N/A";
                            summary.append(String.format("Epoch %d: Avg Reward %.4f, Epsilon %s, Invalid %s%%\n", ep, reward, eps, inv));
                        }
                    }
                } catch (NumberFormatException ignored) {}
            }
            
            if (count == 0) return "No data found in the specified epoch range.";
            
            summary.append(String.format("\nSummary over %d epochs:\n", count));
            summary.append(String.format("Start Reward: %.4f | End Reward: %.4f\n", startReward, endReward));
            summary.append(String.format("Min Reward: %.4f | Max Reward: %.4f\n", minReward, maxReward));
            if (endReward > startReward) summary.append("Trend: IMPROVING\n");
            else summary.append("Trend: STAGNANT/DEGRADING\n");
            
            return summary.toString();
        } catch (Exception e) {
            return "Error reading logs: " + e.getMessage();
        }
    }

    private void startNewRun(SupervisorDecision decision) {
        java.util.function.Consumer<String> cb = this.statusCallback;
        if (rlService.isTrainingRunning()) {
            if (cb != null) {
                cb.accept("[ORCHESTRATOR] Start-run skipped: training is already running.");
            }
            return;
        }

        Boolean use2dCnn = decision.getProposedUse2dCnn();
        boolean is2d = use2dCnn != null ? use2dCnn : false;
        
        RLRewardConfig rewardConfig = decision.getProposedRewardConfig();
        if (rewardConfig != null) {
            rlService.setRewardConfig(rewardConfig);
        }
        
        String modelName = decision.getProposedModelName();
        if (modelName == null || modelName.trim().isEmpty()) {
            modelName = "auto_run_" + System.currentTimeMillis();
        }
        
        LlmSupervisorConfig supervisorConfig = rlService.getSupervisorConfig();
        long durationMs = supervisorConfig.getAutopilotTrainingDurationMs();
        RLTrainingConfig trainingConfig = new RLTrainingConfig(
            modelName.trim(), 100, 40, 1.0, 0.8, 1.2, 1.0, 0.5, 50,
            supervisorConfig, durationMs, is2d);
            
        java.util.function.Consumer<TrainingMetrics> trainingCb = this.trainingProgressCallback;
        if (cb != null) {
            cb.accept("[ORCHESTRATOR] Training started: " + trainingConfig.getModelName());
            if (durationMs > 0) {
                cb.accept("[ORCHESTRATOR] Autopilot run time limit: " + formatDuration(durationMs));
            }
        }

        java.util.function.Consumer<com.rustbuilder.ai.core.TrainingMetrics> progressCb = null;
        if (trainingCb != null) {
            progressCb = trainingCb;
        } else if (cb != null) {
            progressCb = m -> {
                String progress = String.format(
                    "[AUTO] Ep %d/%d Epoch %d/%d | Score %.4f | Best %.4f | Eps %.4f | Loss %.6f | Invalid %.1f%%",
                    m.currentEpisodeInEpoch, m.totalEpisodesPerEpoch,
                    m.currentEpoch, m.totalEpochs,
                    m.currentEpisodeEvalScore, m.bestScore,
                    m.epsilon, m.lastTrainLoss,
                    m.invalidActionRate * 100);
                cb.accept(progress);
            };
        }

        try {
            rlService.train(trainingConfig, progressCb, null);
            if (cb != null) {
                cb.accept("[ORCHESTRATOR] Training finished: " + trainingConfig.getModelName());
            }
        } catch (Throwable e) {
            if (cb != null) {
                java.io.StringWriter sw = new java.io.StringWriter();
                e.printStackTrace(new java.io.PrintWriter(sw));
                cb.accept("[ORCHESTRATOR] Training ERROR: " + sw.toString());
            }
        }
    }

    private static String formatDuration(long ms) {
        long safeMs = Math.max(0L, ms);
        long totalSeconds = safeMs / 1000L;
        long hours = totalSeconds / 3600L;
        long minutes = (totalSeconds % 3600L) / 60L;
        long seconds = totalSeconds % 60L;
        if (hours > 0) {
            return String.format("%dh %02dm", hours, minutes);
        }
        if (minutes > 0) {
            return String.format("%dm %02ds", minutes, seconds);
        }
        return String.format("%ds", seconds);
    }
}
