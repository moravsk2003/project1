# Project Documentation (AI_CONTEXT)

> [!IMPORTANT]
> **SOURCE OF TRUTH**: Код є абсолютним джерелом істини. Будь-яка зовнішня інформація про Rust або припущення з цієї документації є другорядними щодо фактичної імплементації в коді.

## 1. Project Summary
Цей проєкт — симулятор і генератор архітектури баз для гри Rust ("Rust Base Builder"). Основна мета: процедурна генерація оптимальних, стійких до рейду та ефективних за вартістю і логістикою баз.

Проєкт поєднує:
*   **Core Domain**: фізичний рушій на основі 3D-сітки (Grid), що симулює правила будівництва Rust (стабільність, сокети, колізії).
*   **Evaluators**: графова система оцінки (HouseGraph, Dijkstra), що рахує вартість рейду (з урахуванням splash damage) та логістичну зв'язність.
*   **AI Generators**: два ізольовані рушії генерації: Evolutionary Algorithms (EA) та Reinforcement Learning (RL).

## 2. Core Modules (Source of Truth)

### 2.1. com.rustbuilder.model.core & com.rustbuilder.model
*   **GridModel**: Центральне сховище стану. Використовує побітовий зсув (bit-shifting) для генерації 64-бітних ключів просторового індексу (`getSpatialKey`). Це критично для продуктивності методів `hasCollision` та `getNearbyBlocks` ($O(1)$ пошук).
*   **BuildingBlock**: Визначає геометрію та тип блоку. Позиція (X, Y) базується на `TILE_SIZE` (60px), а Z — на висоті поверху.
*   **Socket**: Точки прив'язки для конструкцій. З'єднання вважається успішним, якщо відстань між сокетами $< 1.3$.

### 2.2. com.rustbuilder.service.physics
*   **StabilityService**: Розраховує структурну підтримку. Використовує BFS-поширення підтримки від фундаментів (стабільність 1.0). Блоки зі стабільністю $< 0.1$ автоматично видаляються через `recalculateAll()`.
*   **SnappingService**: Логіка автоматичного вирівнювання блоків у UI-редакторі.

### 2.3. com.rustbuilder.service.evaluator & .graph
*   **HouseGraph**: Будує 3D-граф, де вузли — це тайли, а ребра — переходи через стіни/двері. Вузли для меблів (TC, LootRoom) створюються як дочірні до тайлових вузлів.
*   **RaidResistanceEvaluator**: Алгоритм Дейкстри на графі вартості сірки.
*   **Coverage Heuristic**: Впроваджено метод `calculateLocalCoverage`, який сумує `splashCosts` суміжних стін навколо TC/Loot. Це вирішує проблему "плоского фітнес-ландшафту", надаючи градієнт нагороди за часткову забудову кімнати.

## 3. RL Engine (Multi-Discrete Flow)

### 3.1. Architecture of Actions
*   **MultiDiscreteDQNAgent**: Нейромережа DL4J зі спільним CNN-стовбуром та 5 головами (heads).
*   **Balanced Replay Memory**: У `MultiDiscreteExperienceReplay` впроваджено використання двох ізольованих буферів (`validBuffer` та `invalidBuffer`). Метод `sample()` забезпечує збалансований семплінг (50/50). Це архітектурне рішення запобігає деградації мережі на ранніх етапах, коли агент міг би навчитися лише дії STOP через домінування штрафів за невалідні спроби розміщення.
*   **Phases**:
    1. **Type**: Вибір типу блоку (Foundation, Wall, Floor, etc.) або STOP.
    2. **Floor**: Вибір рівня (0..7).
    3. **Tile**: Вибір позиції на сітці (0..63).
    4. **Rotation**: Орієнтація (0..3).
    5. **Aim**: Тонке позиціонування всередині тайла (0..24).

### 3.2. Heuristic Masking (Phase-by-Phase)
`HeuristicMaskingUtils` мінімізує простір пошуку за допомогою масок:
*   **Phases 1-4**: Використовують "дешеві" евристики (перевірка `hasWallAtFloor` нижче, перевірка `passesNearStructureRule`). Ресурсоємні dry-runs ("continuation search") вилучені.
*   **Phase 5 (Aim Sector)**: ТІЛЬКИ ця фаза виконує точну фізичну перевірку через `GridPlacementUtils.isActionActuallyFeasible`.
*   **Sector Optimization**: Перевірка 25 секторів виконується в порядку `SECTOR_OPTIMIZED_ORDER` (Центр -> Кільце 1 -> Краї), що дозволяє припинити пошук після першого валідного сектора.

### 3.3. Episode Evaluator & Rewards
*   **DFS Connectivity**: Перед видачею нагороди база розбивається на компоненти зв'язності.
*   **fragmentPenalty**: Розраховується як `fragmentBasePenalty + (minDistTiles^2 * fragmentDistPenaltyMult)`.
*   **tcPenalty**: Штраф за ізоляцію TC від основного компонента.
*   **Head-Aware Credit Assignment**: У `MultiDiscreteDQNAgent.trainBatch` реалізовано евристичний розподіл провини між головами мережі. Наприклад, якщо `reward <= penaltyNoSupport`, штраф для голів Rotation та Floor збільшується у 1.5 та 1.2 рази відповідно. Якщо `reward <= penaltyBadSocket`, основний штраф отримують голови Tile та Type.
*   **Reward Shaping**: Динамічна зміна `RLRewardConfig` (понад 20 параметрів) без перезапуску.

## 4. Evolutionary Engine (EA Flow)

### 4.1. Genetic Algorithm Service
*   **Adaptive Mutation**: Збільшення `currentMutationRate` при стагнації понад `stagnationLimit`.
*   **Island Restart**: При стагнації $> stagnationLimit * 2$ нижня половина популяції замінюється новими геномами з гарантованим TC (`randomGenomeWithTC`).
*   **Death Spiral Fix**: Обов'язкове скидання `stagnationCounter = 0` та `currentMutationRate = baseMutationRate` при Island Restart, щоб уникнути нескінченної 100% мутації.

## 5. Runtime Flows

### 5.1. RL Training Loop
1. **Context Init**: Створення `MultiDiscretePhaseContext`.
2. **Feature Extraction**: `StateEncoder` кодує сітку в тензор. Кодування відбувається СУВОРО до будь-яких змін у поточній ітерації.
3. **Autoregressive Selection**: Вибір фази N подається як one-hot на вхід фази N+1.
4. **Placement & Physics**: Застосування через `GridPlacementUtils` та перерахунок `StabilityService`.
5. **Sparse Reward Shaping (Tail Distribution)**: Для вирішення проблеми рідкісних нагород у `EpisodeEvaluator` впроваджено механізм розмазування фінальної оцінки епізоду (`shapedTailReward = 25%` від `finalEvalReward`) на останні 8 кроків агента в пам'яті (backward distribution). Це прискорює конвергенцію та допомагає мережі краще зрозуміти логіку успішного завершення будівництва.

### 5.2. Persistence & Logging
*   **RLModelManager**: Зберігає `.rmeta` (метадані), `.rnet` (мережа), `.rqtb` (табличний бейзлайн QTable).
*   **RLTrainingLogger**: Працює в асинхронному режимі. Генерує 3 файли: `epoch.csv`, `episodes.csv`, `invalid_actions.csv`.

## 6. Domain Glossary
*   **TILE_SIZE**: 60.0 пікселів.
*   **Socket Radius**: 1.3 (максимальна відстань з'єднання).
*   **Stability Threshold**: 0.1 (якщо менше — блок руйнується).
*   **Splash Damage**: 1 ракета пошкоджує 4 суміжні стіни одночасно.
*   **STOP Action**: Перериває будівництво; нараховує `stopUnbuiltBlockPenalty` (-0.01 за кожен недобудований крок).

## 7. Critical Contracts and Invariants
1.  **Маскування ≡ Фізика**: `HeuristicMaskingUtils` ПОВИННА повертати лише те, що пройде `GridPlacementUtils.isActionActuallyFeasible`.
2.  **Stability First**: `StabilityService.recalculateAll()` має бути викликаний ДО подрахунку нагород.
3.  **Action Mapping**: `RotationIndex` (0..3) строго відповідає `Orientation` або градусам.
4.  **Graph Integrity**: Граф будується лише на стабільних блоках після `grid.finalizeLoad()`.
5.  **Death Spiral Prevention**: Завжди скидати лічильники при Island Restart.

## 8. Known Risks and Common Misreadings
*   **God Object**: `RLTrainingService` перевантажений (епізоди, UI, логи).
*   **Credit Assignment Hack**: Евристичний розподіл нагороди між "головами" нейромережі.
*   **Performance**: `HeuristicMaskingUtils` — вузьке місце, якщо не використовувати оптимізований порядок секторів.
*   **MDP Risk**: Параметр `step` вилучений із хешування стану в `QTable` для збереження марковської властивості.

## 9. File Request Strategy for AI
1.  **Не проси все відразу**.
2.  **Починай з інтерфейсів**: `MultiDiscretePhasePolicy`, `MultiDiscreteStateObserver`.
3.  **Ізолюй логіку**: `GridModel` (стан) vs `GridPlacementUtils` (зміна стану).
4.  **Йди за потоком даних**: `NeuralMultiDiscreteDecisionProvider` -> `MultiDiscreteActionMapper` -> `GridPlacementUtils`.

## 10. Task-Oriented File Maps
*   **RL Action Space**: `HeuristicMaskingUtils`, `GridPlacementUtils`, `MultiDiscreteActionMapper`, `NeuralMultiDiscreteDecisionProvider`.
*   **Evaluation & Rewards**: `HouseEvaluator`, `StepRewardFunction`, `RaidResistanceEvaluator`, `EpisodeEvaluator`, `HouseGraph`.
*   **Physics**: `GridModel`, `StabilityService`, `GridPlacementUtils`, `CollisionUtils`.
*   **GA Evolution**: `GeneticAlgorithmService`, `BaseGenome`.

## 11. Safe Refactoring Guidance
*   Зміни у фізиці (сокети, колізії) вимагають синхронних змін у `HeuristicMaskingUtils`.
*   Зміна `HouseGraph` ламає розрахунок рейду (Dijkstra).
*   Не змінюйте розмірності тензорів у `StateEncoder` без оновлення архітектури в `MultiDiscreteDQNAgent`.

## 12. Open Questions / Unstable Areas
*   **Phase 5 (Aim Sector)**: Активне калібрування нагород для стимуляції навчання точного примагнічування.
*   **STOP Logic**: Потребує стабілізації механізму раннього завершення епізоду з бонусом за якість.
*   **QTable Baseline**: Синхронізація табличного бейзлайну з новим 5-фазним простором дій.
*   **Credit Assignment**: Математика перерозподілу штрафів між головами нейромережі залишається експериментальною.
