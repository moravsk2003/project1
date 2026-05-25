package com.rustbuilder.ai.rl.supervisor.application;


import com.rustbuilder.ai.rl.domain.log.StopReason;
import com.rustbuilder.ai.rl.supervisor.config.LlmSupervisorApplyMode;
import com.rustbuilder.ai.rl.supervisor.config.LlmSupervisorConfig;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorAction;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorDecision;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorObservation;
import com.rustbuilder.ai.rl.supervisor.ports.LlmSupervisor;
import com.rustbuilder.ai.rl.supervisor.provider.LlmSupervisorFactory;
import com.rustbuilder.ai.rl.supervisor.validation.SupervisorDecisionValidator;
import com.rustbuilder.ai.core.TrainingMetrics;
import com.rustbuilder.ai.rl.application.RLTrainingConfig;
import com.rustbuilder.ai.rl.application.RLTrainingService;

import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Background orchestrator that autonomously controls the RLTrainingService.
 * When enabled, it periodically polls the LLM when idle to decide whether to start a new run.
 */
public class LlmOrchestrator {

    private final RLTrainingService rlService;
    private final HistoricalTrainingReportService historicalReportService;
    private final SupervisorDecisionValidator decisionValidator = new SupervisorDecisionValidator();
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final AtomicBoolean idleDecisionInProgress = new AtomicBoolean(false);
    private final Object executorLock = new Object();
    private ScheduledExecutorService scheduler;
    private ExecutorService forceCheckExecutor;
    private ScheduledFuture<?> scheduledTask;
    private volatile java.util.function.Consumer<String> statusCallback;
    private volatile java.util.function.Consumer<TrainingMetrics> trainingProgressCallback;
    
    // Interval to poll the LLM when idle (in milliseconds)
    private static final long IDLE_POLL_INTERVAL_MS = 10000;

    public LlmOrchestrator(RLTrainingService rlService) {
        this(rlService, new HistoricalTrainingReportService());
    }

    public LlmOrchestrator(RLTrainingService rlService,
                           HistoricalTrainingReportService historicalReportService) {
        this.rlService = Objects.requireNonNull(rlService, "rlService");
        this.historicalReportService = Objects.requireNonNull(historicalReportService, "historicalReportService");
    }

    public void start() {
        if (running.compareAndSet(false, true)) {
            synchronized (executorLock) {
                scheduler = Executors.newSingleThreadScheduledExecutor(daemonThreadFactory("llm-orchestrator"));
                scheduledTask = scheduler.scheduleWithFixedDelay(
                    this::runScheduledCheck,
                    0L,
                    IDLE_POLL_INTERVAL_MS,
                    TimeUnit.MILLISECONDS);
            }
        }
    }

    public void stop() {
        running.set(false);
        synchronized (executorLock) {
            if (scheduledTask != null) {
                scheduledTask.cancel(true);
                scheduledTask = null;
            }
            if (scheduler != null) {
                scheduler.shutdownNow();
                scheduler = null;
            }
            if (forceCheckExecutor != null) {
                forceCheckExecutor.shutdownNow();
                forceCheckExecutor = null;
            }
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
        try {
            getForceCheckExecutor().execute(() -> {
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
                    if (rlService.isSupervisorBranchWorkInProgress()) {
                        cb.accept("Orchestrator: branch experiment/review is still running.");
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
        } catch (RejectedExecutionException e) {
            idleDecisionInProgress.set(false);
            cb.accept("[ORCHESTRATOR] Error: executor is not accepting checks.");
        }
    }

    private void runScheduledCheck() {
        if (!running.get()) {
            return;
        }
        try {
            LlmSupervisorConfig config = rlService.getSupervisorConfig();
            if (config == null || !config.isEnabled() || rlService.isTrainingRunning()) {
                return;
            }
            if (rlService.isSupervisorBranchWorkInProgress()) {
                return;
            }
            if (!idleDecisionInProgress.compareAndSet(false, true)) {
                return;
            }

            try {
                if (rlService.hasPendingAutoResume()) {
                    rlService.clearPendingAutoResume();
                    String modelName = rlService.getActiveTrainingModelName();
                    if (modelName == null || modelName.isBlank()) modelName = "auto_run_" + System.currentTimeMillis();

                    SupervisorDecision autoResumeDecision = SupervisorDecision.startNewRun(
                        null,
                        null,
                        modelName,
                        "Auto-resuming training after branch jump/promotion."
                    );

                    java.util.function.Consumer<String> cb = msg -> {
                        if (this.statusCallback != null) this.statusCallback.accept(msg);
                    };
                    cb.accept("[ORCHESTRATOR] Auto-resume triggered. Bypassing LLM check.");
                    processDecision(autoResumeDecision, null, config, cb, 0);
                } else {
                    SupervisorObservation obs = rlService.createIdleSupervisorObservation("idle_check");
                    SupervisorDecision decision = askLlm(config, obs);
                    processDecision(decision, obs, config, msg -> {
                        if (this.statusCallback != null) this.statusCallback.accept(msg);
                    }, 0);
                }
            } finally {
                idleDecisionInProgress.set(false);
            }
        } catch (Exception e) {
            // Keep scheduled polling alive through transient supervisor/runtime errors.
            java.util.function.Consumer<String> cb = statusCallback;
            if (cb != null) {
                cb.accept("[ORCHESTRATOR] Scheduled check skipped after error: " + e.getMessage());
            }
        }
    }

    private ExecutorService getForceCheckExecutor() {
        synchronized (executorLock) {
            if (forceCheckExecutor == null || forceCheckExecutor.isShutdown() || forceCheckExecutor.isTerminated()) {
                forceCheckExecutor = Executors.newSingleThreadExecutor(daemonThreadFactory("llm-orchestrator-force-check"));
            }
            return forceCheckExecutor;
        }
    }

    private static ThreadFactory daemonThreadFactory(String name) {
        ThreadFactory baseFactory = Executors.defaultThreadFactory();
        return task -> {
            Thread thread = baseFactory.newThread(task);
            thread.setName(name);
            thread.setDaemon(true);
            return thread;
        };
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
            rlService.recordSupervisorDecisionOutcome(null, "idle");
            cb.accept("[ORCHESTRATOR] LLM returned no decision (check API key / command).");
            rlService.setLastSupervisorDecision(null, "Orchestrator: LLM returned no decision or error occurred.");
            return;
        }
        rlService.recordSupervisorDecisionOutcome(decision, "idle");

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

        if (decision.getAction() == SupervisorAction.START_NEW_RUN
                || decision.getAction() == SupervisorAction.RESTART_TRAINING
                || decision.getAction() == SupervisorAction.LOAD_EXISTING_MODEL) {
            if (config.getApplyMode() != LlmSupervisorApplyMode.AUTO_APPLY) {
                cb.accept("[ORCHESTRATOR] Start-run decision recorded but not applied because apply mode is " + config.getApplyMode() + ".");
                return;
            }
            String modelName = decision.getProposedModelName();
            if (modelName == null || modelName.isBlank()) modelName = "auto_" + System.currentTimeMillis();
            cb.accept(decision.getAction() == SupervisorAction.LOAD_EXISTING_MODEL
                ? "[ORCHESTRATOR] Loading existing model for training: " + modelName.trim()
                : "[ORCHESTRATOR] Starting new run: " + modelName.trim());
            startNewRun(decision);
        } else if (decision.getAction() == SupervisorAction.SET_CURRICULUM_OBJECTIVE) {
            if (config.getApplyMode() != LlmSupervisorApplyMode.AUTO_APPLY) {
                cb.accept("[ORCHESTRATOR] Curriculum objective decision recorded but not applied because apply mode is " + config.getApplyMode() + ".");
                return;
            }
            boolean changed = rlService.applyCurriculumObjective(decision);
            cb.accept(changed
                ? "[ORCHESTRATOR] Curriculum objective handled: " + decision.getObjectiveId()
                : "[ORCHESTRATOR] Curriculum objective change rejected: " + decision.getObjectiveId());
        } else if (decision.getAction() == SupervisorAction.PROMOTE_BRANCH) {
            if (config.getApplyMode() != LlmSupervisorApplyMode.AUTO_APPLY) {
                cb.accept("[ORCHESTRATOR] Promote-branch decision recorded but not applied because apply mode is " + config.getApplyMode() + ".");
                return;
            }
            boolean promoted = rlService.promoteLatestBranch();
            String branchStatus = rlService.getSupervisorBranchStatus();
            cb.accept(promoted
                ? "[ORCHESTRATOR] " + (branchStatus == null || branchStatus.isBlank()
                    ? "Candidate branch promotion handled."
                    : branchStatus)
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
            String report = historicalReportService.generateHistoricalReport(
                decision.getReportModelName(),
                decision.getReportStartEpoch(),
                decision.getReportEndEpoch());
            SupervisorObservation baseObs = originalObs != null
                ? originalObs
                : rlService.createIdleSupervisorObservation("historical_report");
            SupervisorObservation newObs = baseObs.withHistoricalReport(report);
            
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

    private void startNewRun(SupervisorDecision decision) {
        java.util.function.Consumer<String> cb = this.statusCallback;
        if (rlService.isTrainingRunning()) {
            if (cb != null) {
                cb.accept("[ORCHESTRATOR] Start-run skipped: training is already running.");
            }
            return;
        }

        boolean loadExisting = decision.getAction() == SupervisorAction.LOAD_EXISTING_MODEL;
        Boolean use2dCnn = decision.getProposedUse2dCnn();
        boolean is2d = loadExisting ? rlService.isUse2dCnn() : use2dCnn != null ? use2dCnn : false;
        
        String modelName = decision.getProposedModelName();
        if (modelName == null || modelName.trim().isEmpty()) {
            modelName = "auto_run_" + System.currentTimeMillis();
        }
        modelName = modelName.trim();
        
        boolean proposedModelExists = historicalReportService.modelMetadataExists(modelName);

        String activeModel = rlService.getActiveTrainingModelName();

        if (loadExisting) {
            if (!proposedModelExists) {
                if (cb != null) {
                    cb.accept("[ORCHESTRATOR] Load-existing rejected: model not found: " + modelName);
                }
                return;
            }
            if (modelName.equals(activeModel)) {
                if (cb != null) {
                    cb.accept("[ORCHESTRATOR] Resuming training for active model: " + modelName + " (weights already loaded)");
                }
            } else {
                try {
                    rlService.loadExistingModelForTraining(modelName);
                    if (cb != null) {
                        cb.accept("[ORCHESTRATOR] Existing model loaded safely: " + modelName);
                    }
                } catch (Exception e) {
                    if (cb != null) {
                        cb.accept("[ORCHESTRATOR] Failed to load existing model; training not started. Error: " + e.getMessage());
                    }
                    return;
                }
            }
            is2d = rlService.isUse2dCnn();
        } else if (proposedModelExists && modelName.equals(activeModel)) {
            // Best case: the proposed model is exactly the one already loaded in memory
            // (e.g. right after PROMOTE_BRANCH). Skip both reset and load.
            if (cb != null) {
                cb.accept("[ORCHESTRATOR] Resuming training for active model: " + modelName + " (weights already loaded)");
            }
        } else if (proposedModelExists) {
            // The proposed model exists on disk but is not the currently active one — load it.
            if (cb != null) {
                cb.accept("[ORCHESTRATOR] Start-new rejected: model already exists. Use LOAD_EXISTING_MODEL for " + modelName);
            }
            return;
        } else {
            // No model file on disk — truly a fresh start with random Xavier weights.
            rlService.resetForNewTrainingRun(is2d);
        }
        if (decision.getProposedEpsilon() != null) {
            rlService.setEpsilon(clampStartupEpsilon(decision.getProposedEpsilon()));
            if (cb != null) {
                cb.accept(String.format("[ORCHESTRATOR] Startup epsilon set to %.4f", rlService.getEpsilon()));
            }
        }

        LlmSupervisorConfig supervisorConfig = rlService.getSupervisorConfig();
        long durationMs = supervisorConfig.getAutopilotTrainingDurationMs();
        RLTrainingConfig trainingConfig = new RLTrainingConfig(
            modelName, 100, 40, 1.0, 0.8, 1.2, 1.0, 0.5, 50,
            supervisorConfig, durationMs, is2d);
            
        java.util.function.Consumer<TrainingMetrics> trainingCb = this.trainingProgressCallback;
        if (cb != null) {
            cb.accept("[ORCHESTRATOR] Training started: " + trainingConfig.getModelName());
            if (durationMs > 0) {
                cb.accept("[ORCHESTRATOR] Autopilot run time limit: " + formatDuration(durationMs));
            }
        }
        rlService.recordSupervisorAppliedChange(
            loadExisting ? "LOAD_EXISTING_MODEL" : "START_NEW_RUN",
            trainingConfig.getModelName(),
            decision.getReason());

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
        } catch (Exception e) {
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

    private static double clampStartupEpsilon(double epsilon) {
        if (Double.isNaN(epsilon) || Double.isInfinite(epsilon)) {
            return 1.0;
        }
        return Math.max(0.01, Math.min(1.0, epsilon));
    }
}
