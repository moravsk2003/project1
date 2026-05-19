# LLM Supervisor

The RL supervisor is an optional control layer around `RLTrainingService`.
It receives compact training observations every N episodes and can return a
validated structured decision. The default UI interval is 1000 episodes.

Built-in call limits protect the configured API key:

- `10` requests per minute.
- `1,500` requests per day.
- `1,000,000` estimated tokens per minute.

When a limit is reached, training pauses before the next supervisor call instead
of failing the run.

## UI

Open the RL dialog and use the `LLM Supervisor` section:

- `Enable LLM supervisor`: enables the hook between episodes.
- `Call every episodes`: base episode count for supervisor checks. Default:
  `1000`. The LLM must choose the effective future cadence with
  `callFrequency`; this is not a manual UI setting.
- `Run time limit`: optional wall-clock duration for LLM/autopilot-started
  runs only. Examples: `30m`, `2h`, `01:30`. Empty or `0` means no time limit.
  Manual `Train` uses the separate time limit in `Training Parameters`.
- `Apply mode`: `AUTO_APPLY` applies validated decisions; `LOG_ONLY` and
  `MANUAL_APPROVAL` record decisions without changing training.
- `API key`: optional key for built-in Gemini or a custom external command.
  Built-in Gemini uses it directly in Java. External commands receive it as
  `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `LLM_API_KEY`, and `OPENAI_API_KEY`; it is
  not saved in model metadata.
- `Command`: default is `builtin:gemini`, the Java-native Gemini adapter. When
  the supervisor is enabled, an empty command also uses `builtin:gemini`; disable
  the supervisor to use the safe no-op path.
- `Apply Pending`: applies the most recent validated decision when using
  `MANUAL_APPROVAL`.

## Built-In Gemini

Use this command for the default Java implementation:

```text
builtin:gemini
```

It calls `gemma-4-31b-it` first and retries once with
`gemini-3.1-flash-lite` if the primary model fails, times out, returns
empty/non-JSON content, or hits an HTTP error. Override the models with
`GEMINI_MODEL` and `GEMINI_FALLBACK_MODEL`, or set `GEMINI_FALLBACK_MODEL` to
the same value as `GEMINI_MODEL` to disable the fallback.

Older UI configs that still reference `scripts/llm_supervisor_gemini.ps1` or
`scripts/llm_supervisor_gemini.py` are routed to the same Java implementation so
the UI API key is used consistently. Those bundled wrapper scripts were removed;
custom external commands can still be used if you add your own provider.

## External Command Contract

The command receives one JSON object on stdin and writes one JSON object to
stdout. The app does not run the command through a shell; quote paths with
spaces in the command field.

The project no longer ships Python or PowerShell LLM wrappers. For Gemini, use
`builtin:gemini`. For another provider, enter your own command and keep any
provider-specific wrapper outside the default project scripts.

Example stdout:

```json
{
  "action": "SET_EPSILON",
  "epsilon": 0.25,
  "callFrequency": "SOON",
  "reason": "Invalid action rate is stable, reduce exploration."
}
```

Reward config patch with a formula term:

```json
{
  "action": "REPLACE_REWARD_CONFIG",
  "rewardConfig": {
    "blockGrowthReward": 0.08,
    "penaltyCollision": -0.35
  },
  "rewardTerms": [
    {
      "name": "socket_shape",
      "scope": "STEP",
      "expression": "clamp(socket_connections * 0.1, 0, 0.5)"
    }
  ],
  "reason": "Encourage connected growth without changing baseline branch."
}
```

Allowed actions are included in each observation as `allowedActions`.
Observations also include `actionDirections`, a grouped view of the actions
that are possible right now. Built-in Gemini uses two calls: first it chooses one
available direction, then the second call receives only the actions in that
direction and must choose the final action. Final action responses must include
`callFrequency`: `VERY_SOON` (`x0.25`), `SOON` (`x0.5`), `MEDIUM` (`x1`), or
`LONG` (`x2`). Built-in Gemini treats a final response without `callFrequency`
as invalid and retries with the fallback model. The base value is the UI `Call
every episodes` field. Branch experiments use the same effective episode count
before the branch-review LLM call.
During an active training run the hook allows:

- `KEEP_GOING`
- `SET_EPSILON`
- `REPLACE_REWARD_CONFIG`
- `STOP_TRAINING`
- `REQUEST_PROMOTION_CHECK`
- `PROMOTE_BRANCH`
- `JUMP_TO_BRANCH`

When the autopilot is idle, it allows:

- `KEEP_GOING`
- `START_NEW_RUN`
- `LOAD_EXISTING_MODEL`
- `REQUEST_HISTORICAL_REPORT`
- `PROMOTE_BRANCH`
- `JUMP_TO_BRANCH`

Idle observations include `trendMetrics.availableModels`, a compact catalog of
saved RL models with compatibility, metadata, saved epsilon, and recent training
log summaries. Use `START_NEW_RUN` only for a new model name, and use
`LOAD_EXISTING_MODEL` only for a compatible model from that catalog. Both actions
must include an LLM-selected `epsilon`.

`SET_EPSILON` and `REPLACE_REWARD_CONFIG` do not mutate the live training
branch directly. They create a short baseline-vs-candidate branch experiment.
The next observations include `trendMetrics.latestBranchComparison`; return
`PROMOTE_BRANCH` to apply the candidate reward config/epsilon after reviewing
the comparison. The experiment automatically saves the better branch as an
RL model and reports it as `savedWinnerModelName`. Baseline and candidate
branches are both saved. The saved winner is queued/applied automatically; the
live branch is saved as `*_before_jump_*` before any jump.
`JUMP_TO_BRANCH` can switch only to names from
`trendMetrics.knownBranchModelNames`.

New runs use a grouped folder layout:

- `models_rl/<model>/main/`: main branch snapshots and training logs.
- `models_rl/<model>/branches/<branch_model>/`: branch snapshots and branch
  training logs.
- `models_rl/<model>/branches/<branch_model>/inherited_main/`: copied main logs
  from the moment the branch was created.
- `models_rl/<model>/llm/`: LLM decisions and branch comparison logs.

Branch models are persisted as normal `.rmeta`/`.rnet` snapshots with their own
training CSV logs. Branch comparison results are appended under the live model
LLM directory:

- `<model>_branch_experiments.jsonl`
- `<model>_branch_experiments.csv`

The in-memory `branchHistory` sent back to the LLM is only a small recent
context buffer capped at 10 comparisons.

Formula expressions are arithmetic-only and support the variables/functions
listed in `rewardFormulaContract`. Unknown variables and invalid formulas
evaluate to `0`.

Supervisor observations include training time context:

- `trainingStartTime`
- `currentTime`
- `trainingDeadline`
- `trainingElapsedSeconds`
- `trainingRemainingSeconds`
- `trainingTimeLimitEnabled`
- `trainingTimeLimitReached`

They also include:

- `trainingContext`: current epoch/episode, target episodes, max steps,
  encoder mode, 2D/3D CNN flag, epsilon schedule, run/model identifiers, and
  the LLM/autopilot run time limit from the LLM block.
- `trendMetrics`: deltas since the last supervisor check and how long the
  best score has stalled, plus recent supervisor snapshots and the latest
  branch comparison when available.
- `rewardFormulaContract`: allowed STEP/FINAL formula variables and functions.

Each supervisor response is logged in two files under `models_rl`:

- `<model>_supervisor_decisions.jsonl`
- `<model>_supervisor_decisions.csv`

The CSV includes the parsed LLM decision, whether it was applied, the reason,
LLM-selected call frequency, proposed epsilon when present, reward-config-change
flag, key training metrics, and elapsed/remaining training time. It does not
include the API key.
