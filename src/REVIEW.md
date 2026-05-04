# Code Review — src/

## main_engine.cpp

- `L21: 🔴 bug: `feature_id` uninitialized (`for (size_t feature_id;`). Add `= 0`.`
- `L28-32: 🔴 bug: `datapoints` param never assigned to `tree->datapoints`. Search runs on empty data, always returns 0. Assign before calling `solve`.`
- `L20/47: 🟡 risk: float `INFINITY` assigned to `int Loss`. Use `Infinity` from `common.h`.`
- `L114-119: 🔴 bug: phase 2 undo calls `set_misclassified(id, true, tree)` — should be `false` to erase. Points never get un-misclassified on backtrack.`
- `L38: 🔵 nit: `assert(featureId >= 0)` always true, `FeatureId` is unsigned.`

## misclassified_datapoints.cpp

- `L12-13: 🔴 bug: when `misclassified=false`, early-return fires without erasing entry. Undo path is broken.`
- `L10: 🔵 nit: `assert(datapoint >= 0)` always true, `DatapointId` is unsigned.`

## node.h

- `L17-18: 🔴 bug: `DecisionData` holds `NodePtr &left, &right` as refs. If bound to temporaries → dangling. Change to value members `NodePtr left, right`.`

## node_expansion_scheduler_simple.cpp

- `L22-25: 🟡 risk: enforces strict DFS expansion order via assert. Out-of-order call → crash. Document or enforce structurally.`

## build_tree.cpp

- `L13: 🟡 risk: `Threshold threshold` uninitialized. Survives only because `solve_node` overwrites before read. Zero-initialize explicitly.`

## Global

- `NodeId = char` defined in both `node.h` and `node.fwd.h`. Duplicate — remove from `node.h`.
- `#include <cmath>` in `main_engine.cpp` unused. Remove.

claude --resume 9bfa7efd-03e6-4bbf-8706-34120bbc39c8
