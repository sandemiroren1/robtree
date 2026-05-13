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
