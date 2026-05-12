# AI Context Router

Last checked against code: 2026-05-11.

Use this file as a routing index. It is not a full architecture doc. Open the
smallest file set for the task, then follow references from code only.

## Quick Truths

- Code is the source of truth.
- Current placement path is `PlacementService` + `SocketPlacementResolver`.
  `GridPlacementUtils` does not exist in current source.
- `GridModel` owns blocks, spatial lookup, collision/support checks, and
  stability cleanup.
- Manual UI, EA, and RL all converge through `BuildAction` and placement/physics.
- RL and EA are separate engines. Do not mix their state unless the task says so.
- Ignore `models_rl/*.csv` unless the task is about training logs.

## Source Layout

- App/UI: `MainApp.java`, `controller/GameController.java`, `ui/*`
- Domain state: `model/GridModel.java`, `model/core/*`, `model/structure/*`,
  `model/deployable/*`
- Physics: `service/physics/*`
- Evaluation: `service/evaluator/*`, `service/graph/HouseGraph.java`
- RL: `ai/rl/*`, `ai/rl/multidiscrete/*`, `ai/rl/env/*`
- EA: `ai/ea/*`, `ai/core/AIModelManager.java`
- Tests: `src/test/java/com/rustbuilder/*`

## Task Routes

### Manual UI Placement

Open first:
- `controller/GameController.java`
- `ui/GameCanvas.java`
- `service/physics/SnappingService.java`
- `service/physics/PlacementService.java`
- `model/GridModel.java`

Only if needed:
- `service/physics/SocketPlacementResolver.java`
- `util/BlockFactory.java`
- `util/BuildingTypeUtils.java`
- `model/structure/*`
- `model/deployable/*`

### Placement, Sockets, Collision, Stability

Open first:
- `service/physics/PlacementService.java`
- `service/physics/SocketPlacementResolver.java`
- `model/GridModel.java`
- `service/physics/StabilityService.java`

Only if needed:
- `util/CollisionUtils.java`
- `util/SocketGeometryUtils.java`
- `util/SocketCompatibilityUtils.java`
- `model/core/Socket.java`
- `config/GameConstants.java`

Contracts:
- `BuildAction` fields: type, gridX, gridY, floor, orientation, tier, doorType,
  aimSector.
- `PlacementService.calculatePlacement` applies the 4x4 `aimSector` offset and
  resolves sockets.
- `PlacementService.isActionActuallyFeasible` is the final expensive feasibility
  check used by RL aim masking.
- `GridModel.addBlock` recalculates stability immediately.
- `GridModel.addBlockSilent` must be followed by `finalizeLoad`.
- Stability cleanup removes blocks below `0.1`.

### Evaluation / Scoring

Open first:
- `service/evaluator/HouseEvaluator.java`
- `service/graph/HouseGraph.java`
- the named evaluator: `LogisticsEvaluator`, `ResourceCostEvaluator`,
  `RaidResistanceEvaluator`, `WorkingAreaEvaluator`, or `SafeZoneEvaluator`

Only if needed:
- `service/evaluator/RaidConstants.java`
- `model/GridModel.java`
- `model/core/BuildingBlock.java`

Flow:
- `HouseEvaluator.evaluate(GridModel)` builds one `HouseGraph`.
- Logistics, working area, safe zone, and raid use that graph.
- Resource cost is computed after quality scores.

### RL Training / Rewards

Open first:
- `ai/rl/RLTrainingService.java`
- `ai/rl/EpisodeRunner.java`
- `ai/rl/EpisodeEvaluator.java`
- `ai/rl/StepRewardFunction.java`
- `ai/rl/RLRewardConfig.java`

Only if needed:
- `ai/rl/EpisodeResult.java`
- `ai/rl/RLTrainingLogger.java`
- `service/physics/PlacementService.java`
- `service/evaluator/HouseEvaluator.java`

Flow:
- `RLTrainingService.train` creates runner/evaluator.
- `EpisodeRunner` creates a fresh `GridModel`, encodes state, gets a
  multi-discrete action, maps it to `BuildAction`, places it, finalizes
  stability, computes step reward, stores replay, and trains periodically.
- `EpisodeEvaluator` does final scoring for episodes with more than 5 blocks and
  adds terminal shaping to the last episode transition.

### RL Action Space / Masks / Network

Open first:
- `ai/rl/multidiscrete/MultiDiscreteActionSpace.java`
- `ai/rl/multidiscrete/MultiDiscreteAction.java`
- `ai/rl/multidiscrete/MultiDiscreteActionMapper.java`
- `ai/rl/multidiscrete/HeuristicMaskingUtils.java`
- `ai/rl/multidiscrete/NeuralMultiDiscreteDecisionProvider.java`

Network/training bugs:
- `ai/rl/multidiscrete/MultiDiscreteDQNAgent.java`
- `ai/rl/multidiscrete/MultiDiscreteExperienceReplay.java`
- `ai/rl/multidiscrete/MultiDiscreteCreditAssignment.java`
- `ai/rl/multidiscrete/ActionConditioningUtils.java`

Constants:
- Phases: type, floor, tile, rotation, aim.
- Type count 11, STOP index 10, STOP allowed after 5 blocks.
- Floors 8, tiles 64 on 8x8 grid, rotations 6, aim sectors 16.
- `MultiDiscreteActionMapper` normalizes non-triangle rotations to 0..3 and
  triangles to 0..5.
- Aim sector is active: `PlacementService` uses it.

### RL State Encoders

Open first:
- `ai/rl/env/spec/EncodingRuntimeConfig.java`
- `ai/rl/env/spec/StateEncodingSpec.java`
- `ai/rl/env/spec/ActionSpaceSpec.java`
- `ai/rl/env/state/StateRepresentationEncoder.java`

Then one encoder:
- V1: `VoxelV1StateEncoder.java`
- V2: `BucketedVoxelV2StateEncoder.java`
- V3/default: `HybridV3StateEncoder.java` + `GlobalFeatureEncoder.java`

Current specs:
- V1: `Voxel/v1`, 11 channels, no global vector.
- V2: `Voxel/v2`, 16 channels, no global vector.
- V3: `HYBRID_V3_VOXEL_GLOBAL/v3`, 16 channels, 32 global features.
- All current configs: grid 8x8x8, tile count 64, tile mode `LEGACY_64`.

### RL Persistence

Open first:
- `ai/rl/RLModelManager.java`
- `ai/rl/multidiscrete/MultiDiscreteDQNAgent.java`
- `ui/RLGeneratorDialog.java`

Facts:
- Metadata: `models_rl/<name>.rmeta`.
- Network: `models_rl/<name>.rnet` via DL4J `ModelSerializer`.
- UI load order: metadata -> switch encoder -> compatibility check -> weights.
- `restoreFromModel` validates encoder, grid, action, tile mode, and global flags.
- Risk: `RLModelManager.loadModel(name, service)` loads weights before explicit
  compatibility restore; UI uses safer explicit calls.
- Risk: `loadNetworkWeights` silently skips missing `.rnet`.

### EA Generator

Open first:
- `ai/ea/GeneticAlgorithmService.java`
- `ai/ea/BaseGenome.java`
- `ai/core/AIModelManager.java`
- `ui/GeneratorDialog.java`

Only if needed:
- `service/generator/GeneratorService.java`
- `service/evaluator/HouseEvaluator.java`
- `model/GridModel.java`

Facts:
- EA snapshots are `.dat` files in `models/`.
- `AIModelManager` is EA persistence, not RL persistence.

### JavaFX Hints

Open:
- `ui/hints/HintKey.java`
- `ui/hints/HintUtils.java`
- `ui/hints/ContextHintPopup.java`
- caller dialog/canvas that attaches the hint.

## Change Checklists

Add/change a build action:
- `core/action/BuildAction.java`
- `model/core/BuildingType.java`
- `util/BuildingTypeUtils.java`
- `util/BlockFactory.java`
- `service/physics/PlacementService.java`
- `service/physics/SocketPlacementResolver.java`
- `ai/rl/multidiscrete/MultiDiscreteActionSpace.java`
- `ai/rl/multidiscrete/MultiDiscreteActionMapper.java`
- `ai/rl/multidiscrete/HeuristicMaskingUtils.java`
- UI files only if user-facing.

Change geometry/collision/socket behavior:
- `model/structure/*` or `model/deployable/*`
- `util/CollisionUtils.java`
- `util/SocketGeometryUtils.java`
- `util/SocketCompatibilityUtils.java`
- `service/physics/SocketPlacementResolver.java`
- `service/physics/PlacementService.java`
- `model/GridModel.java`
- `service/physics/StabilityService.java`

Change RL tensor/action dimensions:
- `ai/rl/env/spec/EncodingRuntimeConfig.java`
- `ai/rl/env/spec/StateEncodingSpec.java`
- `ai/rl/env/spec/ActionSpaceSpec.java`
- `ai/rl/multidiscrete/MultiDiscreteDQNAgent.java`
- `ai/rl/RLModelManager.java`
- `ui/RLGeneratorDialog.java`

Change score/reward shaping:
- `service/evaluator/HouseEvaluator.java`
- relevant `service/evaluator/*`
- `ai/rl/StepRewardFunction.java`
- `ai/rl/EpisodeEvaluator.java`
- `ai/rl/RLRewardConfig.java`
- `ui/RLGeneratorDialog.java` if exposed in UI.

## Tests By Area

- Grid/placement: `model/GridModelTest.java`
- Stability: `service/StabilityServiceTest.java`
- Snapping: `service/SnappingServiceTest.java`
- Evaluation: `service/HouseEvaluatorTest.java`
- Block prototypes: `model/core/BuildingBlockPrototypeTest.java`
- EA decode: `ai/BaseGenomeDecodeTest.java`
- RL helper rewards: `ai/rl/EpisodeRunnerTest.java`
- RL credit: `ai/rl/multidiscrete/MultiDiscreteCreditAssignmentTest.java`

## Verification Shortcuts

- Docs-only edit: no compile required.
- Localized Java change: run the matching test class first.
- Full Maven compile can be noisy in this repo; use targeted `javac` when the
  changed surface is narrow.
- RL persistence check: verify `.rmeta` and `.rnet` as a pair. `.rnet` is a DL4J
  ZIP containing `configuration.json`, `coefficients.bin`, and usually
  `updaterState.bin`.

## Known Outdated Docs

- `ARCHITECTURE.md` still mentions `GridPlacementUtils`; current code uses
  `PlacementService`.
- Older prose may say aim sector is ignored; current placement uses
  `BuildAction.aimSector`.
- If docs disagree with code, update this router first and cite source files.
