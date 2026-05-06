# Project Documentation (AI_CONTEXT)

> [!IMPORTANT]
> **SOURCE OF TRUTH**: Код є абсолютним джерелом істини. Будь-яка зовнішня інформація про Rust або припущення з цієї документації є другорядними щодо фактичної імплементації в коді.

## 1. Project Summary
Цей проєкт — Java/JavaFX симулятор і генератор архітектури баз для гри Rust ("Rust Base Builder"). Основна мета: процедурна генерація баз, які балансують стійкість до рейду, ціну ресурсів, логістику, корисну захищену площу та safe-zone метрику.

Проєкт поєднує:
*   **Core Domain**: фізичний рушій на основі 3D-сітки (`GridModel`), що симулює правила будівництва Rust у межах локальної моделі: стабільність, сокети, колізії, поверхи, квадратні й трикутні елементи.
*   **Evaluators**: графова система оцінки (`HouseGraph`, Dijkstra/BFS), що рахує вартість рейду з урахуванням splash-adjusted coverage, логістичну зв'язність, вартість, робочу площу та закриті safe-zone тайли.
*   **AI Generators**: два ізольовані рушії генерації: Evolutionary Algorithms (EA) на геномах `BaseGenome` та Reinforcement Learning (RL) на multi-discrete DQN.

## 2. Core Modules (Source of Truth)

### 2.1. com.rustbuilder.model.core & com.rustbuilder.model
*   **GridModel**: Центральне сховище стану. Використовує 64-бітний просторовий ключ (`getSpatialKey`) і власну `LongBlockListMap` для швидкого пошуку сусідів у методах `getNearbyBlocks`, `hasCollision` та `canPlace`. `addBlock` одразу перераховує стабільність, а `addBlockSilent` призначений для bulk-load і потребує фінального `finalizeLoad`.
*   **BuildingBlock**: Базовий клас геометрії, типу, tier, stability, cost і сокетів. Позиція X/Y працює у world-space пікселях із `GameConstants.TILE_SIZE = 60.0`, Z — індекс поверху.
*   **Socket**: Точки прив'язки для конструкцій. Стабільність вважає сокети з'єднаними, якщо квадрат відстані між ними `< 1.3`; UI snapping окремо використовує `GameConstants.SNAP_RADIUS = 45.0`.

### 2.2. com.rustbuilder.service.physics
*   **StabilityService**: Розраховує стабільність блоків (0.0 - 1.0) через поширення від foundation-блоків і перевірки сокетної підтримки. Фактичне видалення блоків зі stability `< 0.1` виконується в `GridModel.updateStability()` та `GridModel.finalizeLoad()`.
*   **SnappingService**: Логіка автоматичного вирівнювання блоків у UI-редакторі. Для foundation/triangle допускає unsnapped старт, для інших типів шукає валідний socket candidate у локальному радіусі.

### 2.3. com.rustbuilder.service.evaluator & .graph
*   **HouseGraph**: Будує 3D-граф, де вузли — `tile`, `outside`, `tc`, `workbench`, `loot_room`, а ребра — відкриті проходи або заблоковані переходи через стіни/двері/стелю. Для великих графів має просторові індекси тайлів і стін.
*   **RaidResistanceEvaluator**: Виконує Dijkstra від `outside` по raid-cost графу. Рахує sulfur до TC, loot-room і workbench-цілей; для scoring також додає локальне покриття навколо важливих вузлів.
*   **Coverage Heuristic**: `calculateLocalCoverage` сумує `splashCosts` суміжних блокерів навколо TC/Loot/Workbench і сусідніх тайлів. Важливо: під час самого Dijkstra traversal використовується edge raid cost, а splash-adjusted coverage додається як окремий градієнт score.

## 3. RL Engine (Multi-Discrete Flow)

### 3.1. Architecture of Actions
*   **MultiDiscreteDQNAgent**: DL4J `ComputationGraph` зі спільним 3D-CNN стовбуром, опційним global-vector входом для Hybrid V3 та 5 autoregressive output heads.
*   **Balanced Replay Memory**: `MultiDiscreteExperienceReplay` тримає два буфери (`validBuffer` та `invalidBuffer`). `sample()` намагається взяти приблизно половину batch із валідних і половину з невалідних transition, з fallback-ом якщо одного типу недостатньо.
*   **Phases**:
    1. **Type**: Вибір типу блоку або STOP (`TYPE_COUNT = 11`, `STOP_TYPE_INDEX = 10`).
    2. **Floor**: Вибір рівня (`FLOOR_COUNT = 8`).
    3. **Tile**: Вибір позиції на сітці 8x8 (`TILE_COUNT = 64`).
    4. **Rotation**: Орієнтація (`ROTATION_COUNT = 6`, 60-degree кроки для triangle-capable placement; квадратні типи часто звужуються до 0 або cardinal 0..3).
    5. **Aim**: Тонке позиціонування всередині тайла (`AIM_SECTOR_COUNT = 16`, 4x4 grid; default sector 5).

### 3.2. Heuristic Masking (Phase-by-Phase)
`HeuristicMaskingUtils` мінімізує простір пошуку за допомогою масок:
*   **Phases 1-4**: Використовують дешеві евристики: старт із foundation/triangle foundation, заборона дубльованих TC/Loot, floor gating, wall/ceiling/furniture tile candidates, rotation gating. Ресурсоємні continuation scans вилучені.
*   **Phase 5 (Aim Sector)**: Тільки ця фаза виконує точну фізичну перевірку через `GridPlacementUtils.isActionActuallyFeasible`.
*   **Sector Optimization**: Перевірка 16 секторів виконується в `SECTOR_OPTIMIZED_ORDER`: центральні 2x2 сектори -> edge-adjacent -> corners. `getFirstValidAimSector` зупиняється на першому валідному секторі.

### 3.3. Episode Evaluator & Rewards
*   **DFS Connectivity**: Перед фінальною нагородою база розбивається на компоненти зв'язності; для великих епізодів використовується spatial neighbor search.
*   **fragmentPenalty**: Розраховується як `fragmentBasePenalty + (minDistTilesSq * fragmentDistPenaltyMult)` для кожного фрагмента поза головною компонентою.
*   **tcPenalty**: Якщо TC існує, блоки поза TC-компонентою отримують `tcConnectivityPenalty`, а всі блоки додатково отримують дистанційний множник `tcDistancePenaltyMult`.
*   **Head-Aware Credit Assignment**: Розподіл провини реалізований у `MultiDiscreteCreditAssignment.forPlacement`, який повертає multipliers по 5 heads залежно від `PlacementError`. `MultiDiscreteDQNAgent.trainBatch` множить reward кожної head на ці multipliers.
*   **Reward Shaping**: `RLRewardConfig` централізує понад 30 reward/penalty параметрів. `RLTrainingService` передає config у agent і дозволяє зберігати/відновлювати його разом із RL snapshot.

### 3.4. State Representation Encoders
Система енкодерів керується `EncodingRuntimeConfig`:
*   **VoxelV1StateEncoder**: Базовий 11-канальний енкодер (`StateEncodingSpec("Voxel", "v1", 11, [8,8,8])`) без global-vector.
*   **BucketedVoxelV2StateEncoder**: 16-канальний енкодер (`"Voxel", "v2"`) із `VoxelAggregationBuffer` і `SKIP_OUT_OF_BOUNDS` coordinate mode.
*   **HybridV3StateEncoder**: Поточний default у `RLTrainingService`: 16-канальний voxel encoder + 32-фічевий global vector (`HYBRID_V3_VOXEL_GLOBAL`, `v3`) через `GlobalFeatureEncoder` / `GlobalFeatureDiagnostics`.

## 4. Evolutionary Engine (EA Flow)

### 4.1. Genetic Algorithm Service
*   **Adaptive Mutation**: `currentMutationRate` рухається від `baseMutationRate` до `maxMutationRate` при стагнації понад `stagnationLimit`, а після покращення або cooldown повертається ближче до базового значення.
*   **Island Restart**: При `stagnationCounter >= stagnationLimit * 2` нижня частина наступного покоління дозаповнюється новими геномами з гарантованим TC (`randomGenomeWithTC`).
*   **Death Spiral Fix**: Після Island Restart `stagnationCounter` скидається в 0, а `currentMutationRate` — до `baseMutationRate`, щоб не застрягнути в режимі надмірної мутації.

## 5. Runtime Flows

### 5.1. RL Training Loop
1. **Context Init**: Створення `MultiDiscretePhaseContext` із grid snapshot, ознаками `hasTC`/`hasLootRoom`, step і maxSteps.
2. **Feature Extraction**: `StateRepresentationEncoder` кодує поточну сітку перед placement; для V3 також формується global-vector. `EncodedState` володіє ND4J arrays і має `close()`.
3. **Autoregressive Selection**: Вибір фази N подається як one-hot conditioning на фази N+1 (`cond_type`, `cond_floor`, `cond_tile`, `cond_rot`).
4. **Placement & Physics**: `GridPlacementUtils` розраховує placement і feasibility, після реального placement `GridModel` перераховує stability; invalid action пишеться в replay із `PlacementError`.
5. **Sparse Reward Shaping (Tail Distribution)**: `EpisodeEvaluator` додає `shapedTailReward = 25%` від фінальної reward без early-stop penalty до останнього transition епізоду. Код не розподіляє це на 8 останніх кроків; backward propagation очікується через DQN discount (`gamma`).

### 5.2. Persistence & Logging
*   **RLModelManager**: Зберігає `.rmeta` metadata і `.rnet` neural network у `models_rl`. Під час `restoreFromModel` суворо перевіряє encoder name/version, voxel channels, grid shape, action dimensions, tile indexing mode і global/object/graph flags.
*   **RLTrainingLogger**: Працює з CSV writers у межах training run. Генерує epoch training CSV (`*_multi_discrete_training.csv` або legacy), `*_episodes.csv`, `*_invalid_actions.csv`, `*_performance_tmp.csv`, а також `*_run_metadata.json`.

## 6. Domain Glossary
*   **TILE_SIZE**: 60.0 пікселів.
*   **Socket Radius**: Для stability socket match використовується squared distance `< 1.3`; для UI snapping — `SNAP_RADIUS = 45.0`.
*   **Stability Threshold**: 0.1 (якщо менше — `GridModel` видаляє блок після перерахунку).
*   **Splash Damage**: `RaidConstants` моделює ракету як 1400 sulfur і групує до 4 суміжних стін у splash-adjusted coverage.
*   **STOP Action**: Перериває будівництво; доступна після `MIN_BLOCKS_BEFORE_STOP = 5` і штрафується через early-stop/underbuild логіку, включно з `stopUnbuiltBlockPenalty = -0.01`.

## 7. Critical Contracts and Invariants
1.  **Маскування ≡ Фізика**: Фінальний Aim mask у `HeuristicMaskingUtils` має бути узгоджений з `GridPlacementUtils.isActionActuallyFeasible`; дешеві маски фаз 1-4 можуть лише звужувати очевидно безглузді кандидати.
2.  **Stability First**: Після фактичної зміни `GridModel` потрібно мати актуальну stability перед оцінкою. `addBlock`, `removeBlock` і `finalizeLoad` уже викликають перерахунок/видалення нестабільних блоків.
3.  **Action Mapping**: `MultiDiscreteActionSpace.rotationIndexToDegrees` мапить 0..5 у 0..300 градусів; `GridPlacementUtils` для квадратних типів фактично бере orientation `% 4`, а для triangle — `% 6`.
4.  **Graph Integrity**: `HouseGraph` має будуватися на фактичному списку блоків після stability cleanup; при bulk decode потрібно не забути `grid.finalizeLoad()`.
5.  **Death Spiral Prevention**: Завжди скидати `stagnationCounter` і `currentMutationRate` при Island Restart.
6.  **State Compatibility**: `RLModelManager.restoreFromModel` не дозволяє відновлювати snapshot, якщо encoder/action/grid/global metadata не збігаються з поточною `EncodingRuntimeConfig`.

## 8. Known Risks and Common Misreadings
*   **Performance**: `HeuristicMaskingUtils` та графові evaluators є гарячими зонами; для великих станів уже є spatial indices і aim-check budget, тому не варто повертати повний continuation search без профілювання.
*   **MDP Risk**: `step` використовується в `MultiDiscretePhaseContext` для gating STOP/типів, але не є voxel-фічею стану. Будь-яке посилення залежності масок від step треба узгоджувати з тим, що бачить DQN.

## 9. File Request Strategy for AI
1.  **Не проси все відразу**.
2.  **Починай з інтерфейсів**: `MultiDiscretePhasePolicy`, `MultiDiscreteStateObserver`, `StateRepresentationEncoder`.
3.  **Ізолюй логіку**: `GridModel` (стан) vs `GridPlacementUtils` (зміна стану).
4.  **Йди за потоком даних**: `NeuralMultiDiscreteDecisionProvider` -> `MultiDiscreteActionMapper` -> `GridPlacementUtils`.

## 10. Task-Oriented File Maps
*   **Core & Physics**: `GridModel`, `StabilityService`, `SnappingService`, `GridPlacementUtils`, `CollisionUtils`, `BuildingTypeUtils`, `BlockFactory`.
*   **RL Action Space**: `MultiDiscreteActionSpace`, `HeuristicMaskingUtils`, `MultiDiscreteActionMapper`, `NeuralMultiDiscreteDecisionProvider`, `MultiDiscreteCreditAssignment`.
*   **State Encoding**: `EncodingRuntimeConfig`, `StateEncodingSpec`, `VoxelV1StateEncoder`, `BucketedVoxelV2StateEncoder`, `HybridV3StateEncoder`, `GlobalFeatureEncoder`.
*   **Evaluation, Logistics & Logging**: `HouseEvaluator`, `StepRewardFunction`, `RaidResistanceEvaluator`, `LogisticsEvaluator`, `WorkingAreaEvaluator`, `SafeZoneEvaluator`, `EpisodeEvaluator`, `EpisodeRunner`, `HouseGraph`, `RLTrainingLogger`.

## 11. Safe Refactoring Guidance
*   Зміни у фізиці, сокетах або placement повинні синхронно перевіряти `GridPlacementUtils`, `GridModel.canPlace`, `StabilityService` і фінальний Aim mask у `HeuristicMaskingUtils`.
*   Зміна `HouseGraph` впливає на raid, logistics, working area і safe-zone evaluators, тому потребує перевірки всіх чотирьох метрик.
*   Не змінюйте розмірності tensor/global-vector/action heads без оновлення `EncodingRuntimeConfig`, `StateEncodingSpec`, `MultiDiscreteDQNAgent` і compatibility metadata в `RLModelManager`.

## 12. Open Questions / Unstable Areas
*   **Phase 5 (Aim Sector)**: Активне калібрування 4x4 aim sectors і `useAimSectorLearning`; за default neural provider може підставляти перший валідний сектор замість навчання aim head.
*   **Credit Assignment**: Мапінг `PlacementError` -> head multipliers залишається евристичним і потребує емпіричної перевірки на навчанні.
