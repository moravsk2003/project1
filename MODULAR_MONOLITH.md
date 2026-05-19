# Modular Monolith Boundaries

This project is kept as one deployable Java application, but its code is
organized around explicit module boundaries. The goal is local modularity
without the operational cost of microservices.

## Modules

- `model`: building grid, block entities, sockets, and stability rules.
- `service.physics`: placement and snapping workflows that operate on the model.
- `service.graph`: navigation and raid graph construction.
- `service.evaluator`: scoring criteria and final house evaluation.
- `service.raid`: shared raid-cost rules used by graph and evaluator code.
- `ai`: generation strategies. `ai.rl` and `ai.ea` are independent approaches.
  - `ai.ea.application`: EA use cases.
  - `ai.ea.domain`: EA genome/domain model.
  - `ai.rl.application`: RL training orchestration, runners, config, and logs.
  - `ai.rl.domain`: episode results, reward config, reward logic, and formula evaluation.
  - `ai.rl.environment`: state/action specs and state encoders.
  - `ai.rl.infrastructure`: model persistence and default adapter factories.
  - `ai.rl.policy`: action-selection policies such as multi-discrete DQN.
  - `ai.rl.ports`: factory interfaces consumed by the application layer.
  - `ai.rl.supervisor`: LLM supervisor boundary split into application, config, domain, observability, ports, provider, serialization, and validation packages.
- `controller` and `ui`: JavaFX interaction and presentation.
- `core`: shared command/result types used across modules.
- `config` and `util`: low-level shared support.

## Dependency Rules

- The model module must not call application, service, AI, controller, or UI code.
- Services must not depend on AI, controller, or UI code.
- AI modules must not depend on controller or UI code.
- `ai.rl` and `ai.ea` must not call each other directly.
- `service.graph` must not depend on `service.evaluator`; shared raid values live in `service.raid`.

These rules are enforced by `ArchitectureBoundaryTest`.
