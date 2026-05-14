#include "common.h"
#include "main_engine.h"
#include <catch2/catch_test_macros.hpp>
static Configuration make_config(size_t n_features, float perturbation = 0.0f) {
  Configuration config;
  config.number_of_features = n_features;
  config.perturbations_per_feature.assign(n_features,
                                          {perturbation, perturbation});
  config.epsilons_per_feature.assign(n_features, {0.0f, 0.0f});
  return config;
}

// ── basic sanity
// ──────────────────────────────────────────────────────────────

TEST_CASE("solve: empty dataset → loss = 0", "[solver]") {
  Datapoints datapoints;
  Solver solver(make_config(2));
  auto res = solver.solve(datapoints, 2);
  REQUIRE(res == 0);
}

TEST_CASE("solve: loss is non-negative", "[solver]") {
  Datapoints datapoints = {
      {{0.0f, 0.0f}, 0, false},
      {{1.0f, 1.0f}, 1, true},
  };
  Solver solver(make_config(2));
  REQUIRE(solver.solve(datapoints, 2) >= 0);
}

TEST_CASE("solve: loss never exceeds dataset size", "[solver]") {
  Datapoints datapoints = {
      {{0.0f}, 0, false},
      {{0.5f}, 1, true},
      {{1.0f}, 2, false},
  };
  Solver solver(make_config(1));
  Loss loss = solver.solve(datapoints, 2);
  REQUIRE(loss <= static_cast<Loss>(datapoints.size()));
}

// ── zero-loss cases
// ───────────────────────────────────────────────────────────

TEST_CASE("solve: all same class → loss = 0", "[solver]") {
  Datapoints datapoints = {
      {{0.1f}, 0, false},
      {{0.5f}, 1, false},
      {{0.9f}, 2, false},
  };
  Solver solver(make_config(1));
  REQUIRE(solver.solve(datapoints, 1) == 0);
}

TEST_CASE("solve: linearly separable 1D → loss = 0 at depth 1", "[solver]") {
  // x < 0.5 → class 0, x ≥ 0.5 → class 1
  Datapoints datapoints = {
      {{0.1f}, 0, false},
      {{0.2f}, 1, false},
      {{0.7f}, 2, true},
      {{0.9f}, 3, true},
  };
  Solver solver(make_config(1));
  REQUIRE(solver.solve(datapoints, 1) == 0);
}

TEST_CASE("solve: XOR n=4 depth=2 → loss = 0", "[solver]") {
  // XOR is not linearly separable but depth-2 tree can split it perfectly
  Datapoints datapoints = {
      {{0.0f, 0.0f}, 0, false},
      {{1.0f, 0.0f}, 1, true},
      {{0.0f, 1.0f}, 2, true},
      {{1.0f, 1.0f}, 3, false},
  };
  Solver solver(make_config(2));
  REQUIRE(solver.solve(datapoints, 2) == 0);
}

// ── non-zero loss cases
// ───────────────────────────────────────────────────────

TEST_CASE("solve: XOR n=4 depth=1 → loss > 0", "[solver]") {
  // No single axis-aligned split can separate XOR
  Datapoints datapoints = {
      {{0.0f, 0.0f}, 0, false},
      {{1.0f, 0.0f}, 1, true},
      {{0.0f, 1.0f}, 2, true},
      {{1.0f, 1.0f}, 3, false},
  };
  Solver solver(make_config(2));
  REQUIRE(solver.solve(datapoints, 1) > 0);
}

TEST_CASE("solve: two identical points with different labels → loss = 1",
          "[solver]") {
  // One point must always be misclassified
  Datapoints datapoints = {
      {{0.5f}, 0, false},
      {{0.5f}, 1, true},
  };
  Solver solver(make_config(1));
  REQUIRE(solver.solve(datapoints, 2) == 1);
}

// ── perturbation
// ──────────────────────────────────────────────────────────────

TEST_CASE("solve: with perturbation, loss ≥ loss without perturbation",
          "[solver]") {
  Datapoints datapoints = {
      {{0.1f}, 0, false},
      {{0.9f}, 1, true},
  };
  Solver solver_clean(make_config(1, 0.0f));
  Solver solver_perturbed(make_config(1, 0.3f));

  Loss clean = solver_clean.solve(datapoints, 1);
  Loss perturbed = solver_perturbed.solve(datapoints, 1);
  REQUIRE(perturbed >= clean);
}

// ── depth=0 behaviour ────────────────────────────────────────────────────

TEST_CASE("solve: depth=0, all false → loss = 0", "[solver]") {
  Datapoints datapoints = {
      {{0.1f}, 0, false},
      {{0.5f}, 1, false},
      {{0.9f}, 2, false},
  };
  Solver solver(make_config(1));
  REQUIRE(solver.solve(datapoints, 0) == 0);
}

TEST_CASE("solve: depth=0, all true → loss = 0", "[solver]") {
  Datapoints datapoints = {
      {{0.1f}, 0, true},
      {{0.5f}, 1, true},
  };
  Solver solver(make_config(1));
  REQUIRE(solver.solve(datapoints, 0) == 0);
}

TEST_CASE("solve: depth=0, mixed → loss = min(pos, neg)", "[solver]") {
  // 2 false, 3 true → leaf predicts true, misclassifies 2 false points
  Datapoints datapoints = {
      {{0.0f}, 0, false},
      {{0.5f}, 2, true},
      {{0.6f}, 3, true},
      {{0.7f}, 4, true},
  };
  Solver solver(make_config(1));
  REQUIRE(solver.solve(datapoints, 0) == 1);
}

TEST_CASE("solve: single point → loss = 0 at any depth", "[solver]") {
  Datapoints datapoints = {{{0.5f}, 0, true}};
  Solver solver(make_config(1));
  REQUIRE(solver.solve(datapoints, 0) == 0);
  REQUIRE(solver.solve(datapoints, 1) == 0);
  REQUIRE(solver.solve(datapoints, 2) == 0);
}

// ── monotonicity ─────────────────────────────────────────────────────────

TEST_CASE("solve: depth monotonicity — more depth never increases loss",
          "[solver]") {
  // XOR: depth=1 > 0, depth=2 = 0. d+1 ≤ d always.
  Datapoints datapoints = {
      {{0.0f, 0.0f}, 0, false},
      {{1.0f, 0.0f}, 1, true},
      {{0.0f, 1.0f}, 2, true},
      {{1.0f, 1.0f}, 3, false},
  };
  Solver solver(make_config(2));
  Loss d0 = solver.solve(datapoints, 0);
  Loss d1 = solver.solve(datapoints, 1);
  Loss d2 = solver.solve(datapoints, 2);
  REQUIRE(d1 <= d0);
  REQUIRE(d2 <= d1);
}

// ── perturbation semantics ────────────────────────────────────────────────

TEST_CASE("solve: large perturbation makes linearly separable dataset lose",
          "[solver]") {
  // No perturbation: x=0.1(F), x=0.9(T) separable at depth=1 → loss=0
  // With delta=0.5: intervals [-0.4,0.6] and [0.4,1.4] overlap → loss=1
  Datapoints datapoints = {
      {{0.1f}, 0, false},
      {{0.9f}, 1, true},
  };
  Solver clean(make_config(1, 0.0f));
  Solver perturbed(make_config(1, 0.5f));
  REQUIRE(clean.solve(datapoints, 1) == 0);
  REQUIRE(perturbed.solve(datapoints, 1) == 1);
}

// ── multi-feature ─────────────────────────────────────────────────────────

TEST_CASE("solve: 2D dataset separable only on feature 1 → loss = 0 at depth 1",
          "[solver]") {
  // Feature 0 identical across all points (no split power there)
  // Feature 1 cleanly separates: <0.5 → false, ≥0.5 → true
  Datapoints datapoints = {
      {{0.5f, 0.1f}, 0, false},
      {{0.5f, 0.2f}, 1, false},
      {{0.5f, 0.8f}, 2, true},
      {{0.5f, 0.9f}, 3, true},
  };
  Solver solver(make_config(2));
  REQUIRE(solver.solve(datapoints, 1) == 0);
}

// ── loss bounds ───────────────────────────────────────────────────────────

TEST_CASE(
    "solve: loss ≤ min(pos, neg) — always at least as good as majority vote",
    "[solver]") {
  // 3 false, 2 true → majority vote gives 2 errors
  Datapoints datapoints = {
      {{0.0f}, 0, false}, {{0.2f}, 1, false}, {{0.4f}, 2, false},
      {{0.6f}, 3, true},  {{0.8f}, 4, true},
  };
  Solver solver(make_config(1));
  Loss loss = solver.solve(datapoints, 2);
  REQUIRE(loss <= 2);
}

// ── larger datasets ───────────────────────────────────────────────────────

TEST_CASE("solve: 10-point 1D clean separable → loss = 0 at depth 1",
          "[solver]") {
  Datapoints datapoints = {
      {{0.05f}, 0, false}, {{0.15f}, 1, false}, {{0.25f}, 2, false},
      {{0.35f}, 3, false}, {{0.45f}, 4, false}, {{0.55f}, 5, true},
      {{0.65f}, 6, true},  {{0.75f}, 7, true},  {{0.85f}, 8, true},
      {{0.95f}, 9, true},
  };
  Solver solver(make_config(1));
  REQUIRE(solver.solve(datapoints, 1) == 0);
}

TEST_CASE("solve: depth=0, 8 false + 12 true → loss = 8 (majority class wins)",
          "[solver]") {
  Datapoints datapoints;
  for (int i = 0; i < 8; i++)
    datapoints.push_back({{float(i) * 0.01f}, DatapointId(i), false});
  for (int i = 0; i < 12; i++)
    datapoints.push_back({{0.5f + float(i) * 0.01f}, DatapointId(8 + i), true});
  Solver solver(make_config(1));
  REQUIRE(solver.solve(datapoints, 0) == 8);
}

TEST_CASE("solve: 1D valley — false on edges, true in middle; depth=1 loss=3, "
          "depth=2 loss=0",
          "[solver]") {
  // No axis-aligned split can isolate the middle cluster from both edge groups.
  // Every threshold leaves ≥ 3 misclassified at depth=1.
  // A two-level tree can first peel off the left edge group, then split the
  // rest.
  Datapoints datapoints = {
      {{0.1f}, 0, false}, {{0.2f}, 1, false}, {{0.3f}, 2, false},
      {{0.4f}, 3, true},  {{0.5f}, 4, true},  {{0.6f}, 5, true},
      {{0.7f}, 6, false}, {{0.8f}, 7, false}, {{0.9f}, 8, false},
  };
  Solver solver(make_config(1));
  REQUIRE(solver.solve(datapoints, 1) == 3);
  REQUIRE(solver.solve(datapoints, 2) == 0);
}

TEST_CASE("solve: 8-point 2D XOR with 2 points per quadrant; depth=1 loss=4, "
          "depth=2 loss=0",
          "[solver]") {
  // Each axis-aligned depth-1 split bisects the plane into two quadrant-pairs,
  // each containing 2F+2T — minimum 4 misclassified. Depth-2 splits each half
  // cleanly along the other axis.
  Datapoints datapoints = {
      {{0.2f, 0.2f}, 0, false}, {{0.3f, 0.3f}, 1, false}, // bottom-left: F
      {{0.2f, 0.7f}, 2, true},  {{0.3f, 0.8f}, 3, true},  // top-left:    T
      {{0.7f, 0.2f}, 4, true},  {{0.8f, 0.3f}, 5, true},  // bottom-right: T
      {{0.7f, 0.7f}, 6, false}, {{0.8f, 0.8f}, 7, false}, // top-right:   F
  };
  Solver solver(make_config(2));
  REQUIRE(solver.solve(datapoints, 1) == 4);
  REQUIRE(solver.solve(datapoints, 2) == 0);
}

TEST_CASE("solve: identical feature values, mixed labels — depth never helps",
          "[solver]") {
  // All 6 points share x=0.5. Every threshold either sends all to one side
  // or the other; the leaf always sees 4F+2T. Loss = min(4,2) = 2 at any depth.
  Datapoints datapoints = {
      {{0.5f}, 0, false}, {{0.5f}, 1, false}, {{0.5f}, 2, false},
      {{0.5f}, 3, false}, {{0.5f}, 4, true},  {{0.5f}, 5, true},
  };
  Solver solver(make_config(1));
  REQUIRE(solver.solve(datapoints, 0) == 2);
  REQUIRE(solver.solve(datapoints, 1) == 2);
  REQUIRE(solver.solve(datapoints, 2) == 2);
}

TEST_CASE("solve: 3-feature dataset separable only on feature 2 → loss = 0 at "
          "depth 1",
          "[solver]") {
  // Features 0 and 1 are constant (no split power). Feature 2 cleanly
  // separates.
  Datapoints datapoints = {
      {{0.5f, 0.5f, 0.1f}, 0, false}, {{0.5f, 0.5f, 0.2f}, 1, false},
      {{0.5f, 0.5f, 0.3f}, 2, false}, {{0.5f, 0.5f, 0.7f}, 3, true},
      {{0.5f, 0.5f, 0.8f}, 4, true},  {{0.5f, 0.5f, 0.9f}, 5, true},
  };
  Solver solver(make_config(3));
  REQUIRE(solver.solve(datapoints, 1) == 0);
}

TEST_CASE(
    "solve: 10-point separable with small perturbation still achieves loss=0",
    "[solver]") {
  // Gap between last-false (0.45) and first-true (0.55) is 0.1.
  // Perturbation 0.04: max perturbed false = 0.49, min perturbed true = 0.51 →
  // gap remains.
  Datapoints datapoints = {
      {{0.05f}, 0, false}, {{0.15f}, 1, false}, {{0.25f}, 2, false},
      {{0.35f}, 3, false}, {{0.45f}, 4, false}, {{0.55f}, 5, true},
      {{0.65f}, 6, true},  {{0.75f}, 7, true},  {{0.85f}, 8, true},
      {{0.95f}, 9, true},
  };
  Solver solver(make_config(1, 0.04f));
  REQUIRE(solver.solve(datapoints, 1) == 0);
}

TEST_CASE("solve: perturbation that closes the gap increases loss",
          "[solver]") {
  // Same dataset. Perturbation 0.06: 0.45+0.06=0.51 > 0.55-0.06=0.49 →
  // intervals overlap. The overlapping false/true pair forces ≥ 1 error at any
  // split.
  Datapoints datapoints = {
      {{0.05f}, 0, false}, {{0.15f}, 1, false}, {{0.25f}, 2, false},
      {{0.35f}, 3, false}, {{0.45f}, 4, false}, {{0.55f}, 5, true},
      {{0.65f}, 6, true},  {{0.75f}, 7, true},  {{0.85f}, 8, true},
      {{0.95f}, 9, true},
  };
  Solver clean(make_config(1, 0.0f));
  Solver perturbed(make_config(1, 0.06f));
  REQUIRE(clean.solve(datapoints, 1) == 0);
  REQUIRE(perturbed.solve(datapoints, 1) >= 1);
}
