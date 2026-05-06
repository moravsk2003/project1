# Project Architecture

## 1. Purpose
Проєкт є Java-застосунком для проектування, оцінки та автоматичної генерації баз для гри Rust ("Rust Base Builder"). Він поєднує ручний UI-редактор із двома незалежними AI-системами: еволюційними алгоритмами (EA) та навчанням з підкріпленням (RL). Згенеровані бази оцінюються за п'ятьма критеріями: вартість ресурсів, стійкість до рейду, логістика (навігація), робоча зона (Working Area) та безпечна зона (Safe Zone).

## 2. Main Flow (RL Multi-Discrete Training)
Основний сценарій машинного навчання (наразі активний "Multi-Discrete" flow) побудований на авторегресивному принципі прийняття рішень:
1.  **Ініціалізація**: `RLTrainingService` готує середовище (`GridModel`).
2.  **State Encoding**: Стан сітки конвертується в 3D-тензор [1, 11, 8, 8, 8] через `StateEncoder`.
3.  **Евристичне маскування**: `HeuristicMaskingUtils` генерує маски для кожної з 5 фаз, щоб відсікти фізично неможливі дії.
4.  **Auto-regressive Decision**: `MultiDiscreteDQNAgent` послідовно передбачає Type, Floor, Tile, Rotation та Aim. Кожен вибір подається як вхідний сигнал (conditioning) для наступної фази.
5.  **Трансляція**: `MultiDiscreteActionMapper` перетворює 5-фазну дію у об'єкт `BuildAction`.
6.  **Застосування**: `GridPlacementUtils` виконує розміщення з урахуванням сокетів.
7.  **Оновлення фізики**: `StabilityService` перераховує підтримку та видаляє нестабільні блоки.
8.  **Нагорода**: `StepRewardFunction` (на кожному кроці) та `EpisodeEvaluator` (в кінці епізоду) розраховують Q-сигнал.

## 3. Key Modules & Patterns

### 3.1. Domain Layer (Model)
*   **GridModel**: Використовує паттерн "Spatial Index" на основі 64-бітних ключів (bit-shifting) для швидкої перевірки колізій.
*   **BuildingBlock Entity**: Реалізує складну ієрархію від фундаментів до меблів з підтримкою Tier-системи (Wood, Stone, Metal, HQM).
*   **Strict Geometry Engine**: Процедурна генерація трикутних фундаментів та перекриттів (`TriangleFoundation`, `TriangleFloor`) спирається на сувору геометрію: полігони колізій завжди генеруються з масиву рівно 3 вершин. Це гарантує стабільність роботи алгоритму Роздільної Вісі (SAT) в `CollisionUtils` без артефактів.

### 3.2. Physical Engine (Service Physics)
*   **StabilityService**: Реалізує алгоритм BFS для поширення структурної цілісності від опорних точок (фундаментів).
*   **Оптимізація BFS-обходу**: В результаті рефакторингу алгоритм `recalculateAll()` було переведено з $O(N^2)$ (порівняння кожного з кожним) на $O(N \cdot K)$, де $K$ — локальна кількість сусідів. Це стало можливим завдяки використанню 64-бітного просторового індексу (`spatialMap`) у `GridModel.getNearbyBlocks($O(1)$)`. Це критично для швидкодії під час масової генерації баз та навчання ШІ.
*   **CollisionUtils**: Використовує теорему про роздільну вісь (SAT) для точного виявлення перетинів блоків без створення зайвих об'єктів (Zero-GC optimization).

### 3.3. Evaluation Engine (Service Evaluator)
*   **HouseGraph**: Паттерн "Graph Transformation". Перетворює 3D-сітку на навігаційний граф для пошуку шляхів (Dijkstra).
*   **RaidResistanceEvaluator**: Використовує `Coverage Heuristic` для оцінки часткового захисту TC.
*   **EpisodeEvaluator**: Реалізує аналіз зв'язності через пошук у глибину (DFS) для нарахування штрафів за розірвану структуру.

### 3.4. AI Core (RL & EA)
*   **Reinforcement Learning**: DL4J-екосистема з підтримкою `Experience Replay` та `Target Network` синхронізації.
*   **Evolutionary Algorithms**: Реалізує `GeneticAlgorithmService` з адаптивною мутацією та механізмом `Island Restart`.

## 4. Architectural Innovations

### 4.1. Multi-Discrete Masking Strategy
Для оптимізації навчання впроваджено розділення маскування на "дешеве" та "дороге":
*   **Phases 1-4 (Heuristics)**: Використовуються швидкі перевірки без мутації стану (напр. наявність стіни нижче для встановлення підлоги).
*   **Phase 5 (Physical)**: Виконується повна симуляція розміщення (`isActionActuallyFeasible`). 
*   **Sector Optimization**: Перевірка секторів у порядку `12 -> [сусідні] -> [краї]` суттєво зменшує кількість викликів фізичного рушія.

### 4.2. Coverage Heuristic Logic
Для вирішення проблеми нульового градієнта на початкових етапах побудови:
*   **Splash Costs Grouping**: Стіни, що ділять спільну точку, групуються.
*   **Coverage Calculation**: 
    $S_{coverage} = \sum (Cost_{block} \times 0.25)$ для всіх блоків навколо TC.
*   Це дозволяє AI "відчувати" наближення до створення закритої кімнати.

### 4.3. DFS Connectivity Analysis
Кожна база в кінці епізоду RL або покоління EA перевіряється на цілісність:
*   **Components Separation**: База розбивається на ізольовані острови.
*   **Fragment Penalty Formula**: 
    $P_{frag} = Penalty_{base} + (Distance_{tiles}^2 \times Penalty_{mult})$
*   Це змушує AI будувати єдину структуру, а не розкидані блоки.

### 4.4. GA Death Spiral Prevention
Впроваджено паттерн "Reset on Stagnation":
*   Якщо популяція не покращується протягом `stagnationLimit * 2`, спрацьовує `Island Restart`.
*   **Reset Logic**: Примусове скидання `stagnationCounter = 0` та `currentMutationRate = baseMutationRate`.
*   Без цього скидання висока мутація (100%) знищувала б будь-які вдалі знахідки, унеможливлюючи вихід із локального мінімуму.

## 5. UI & Diagnostics System

### 5.1. Diagnostic Observation Pattern
Через інтерфейс `MultiDiscreteStateObserver` реалізовано:
*   **Step-by-step Logging**: Візуалізація вибору на кожній фазі в реальному часі.
*   **Neural Diagnostics**: Використання `runDiagnosticStep` для налагодження масок та ваг мережі.

### 5.2. Dynamic Configuration Management
Клас `RLRewardConfig` (Serializable) дозволяє:
*   Керувати понад 20 параметрами (нагороди за сокети, штрафи за колізії, бонуси за TC).
*   Зберігати та завантажувати конфігурацію разом із моделлю (`.rmeta`).

### 5.3. Contextual Hint Subsystem
Впроваджено патерн відокремленої довідкової логіки: класи `HintUtils` та `ContextHintPopup` інкапсулюють прив'язку підказок (enum `HintKey`) до компонентів JavaFX. Це розвантажує бізнес-логіку контролерів від роботи з UI-подіями та забезпечує уніфікований UX для складних RL/EA параметрів.

## 6. Persistence Model
Система збереження моделей (`RLModelManager`) керує трьома типами артефактів:
1.  **Metadata (.rmeta)**: Стан тренування, ваги нагород, лічильники епізодів.
2.  **Neural Weights (.rnet)**: Ваги нейромережі DeepLearning4J.
3.  **Baseline QTable (.rqtb)**: Табличний бейзлайн для порівняння якості навчання.

## 7. How to Work With This Codebase
*   **Навігація**: UI-логіка (`com.rustbuilder.ui`), Фізика (`com.rustbuilder.service.physics`), ШІ (`com.rustbuilder.ai`).
*   **Додавання нових дій**: Вимагає синхронного оновлення `MultiDiscreteActionSpace`, `MultiDiscreteActionMapper` та `HeuristicMaskingUtils`.
*   **Тюнінг ШІ**: Використовуйте вкладку нагород у `RLGeneratorDialog`. Зміни застосовуються миттєво для наступного епізоду.
*   **Ізоляція**: Не намагайтеся викликати логіку `ai.ea` з `ai.rl` (і навпаки) — це паралельні підходи до генерації.
