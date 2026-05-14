#include "intersection.h"
#include <catch2/catch_test_macros.hpp>

static Configuration make_config(float left_pert, float right_pert) {
  Configuration config;
  config.number_of_features = 1;
  config.perturbations_per_feature = {{left_pert, right_pert}};
  config.epsilons_per_feature = {{0.0f, 0.0f}};
  return config;
}

static Datapoint make_point(float x) { return {{x}, 0, false}; }

TEST_CASE("Intersection: point below threshold, no perturbation → left only",
          "[intersection]") {
  auto config = make_config(0.0f, 0.0f);
  Threshold t{0, 0.5f};
  auto point = make_point(0.2f);
  auto result = Intersection::intersect_point_with_threshold(t, point, config);
  REQUIRE(result.flows_to_left_subtree);
  REQUIRE_FALSE(result.flows_to_right_subtree);
}

TEST_CASE("Intersection: point above threshold, no perturbation → right only",
          "[intersection]") {
  auto config = make_config(0.0f, 0.0f);
  Threshold t{0, 0.5f};
  auto point = make_point(0.8f);
  auto result = Intersection::intersect_point_with_threshold(t, point, config);
  REQUIRE_FALSE(result.flows_to_left_subtree);
  REQUIRE(result.flows_to_right_subtree);
}

TEST_CASE(
    "Intersection: point exactly at threshold, no perturbation → left only",
    "[intersection]") {
  auto config = make_config(0.0f, 0.0f);
  Threshold t{0, 0.5f};
  auto point = make_point(0.5f);
  auto result = Intersection::intersect_point_with_threshold(t, point, config);
  REQUIRE(result.flows_to_left_subtree);
  REQUIRE_FALSE(result.flows_to_right_subtree);
}

TEST_CASE("Intersection: perturbation pushes point into both subtrees",
          "[intersection]") {
  // x=0.45, delta_R=0.1 → 0.45+0.1=0.55 >= 0.5 (right)
  // x=0.45, delta_L=0.1 → 0.45-0.1=0.35 <= 0.5 (left)
  auto config = make_config(0.1f, 0.1f);
  Threshold t{0, 0.5f};
  auto point = make_point(0.45f);
  auto result = Intersection::intersect_point_with_threshold(t, point, config);
  REQUIRE(result.flows_to_left_subtree);
  REQUIRE(result.flows_to_right_subtree);
}

TEST_CASE(
    "Intersection: large perturbation, point far left still reaches right",
    "[intersection]") {
  auto config = make_config(0.0f, 1.0f);
  Threshold t{0, 0.5f};
  auto point =
      make_point(0.0f); // 0.0 + 1.0 >= 0.5 → right; 0.0 - 0.0 <= 0.5 → left
  auto result = Intersection::intersect_point_with_threshold(t, point, config);
  REQUIRE(result.flows_to_left_subtree);
  REQUIRE(result.flows_to_right_subtree);
}

TEST_CASE(
    "Intersection: point far above threshold with no perturbation → right only",
    "[intersection]") {
  auto config = make_config(0.0f, 0.0f);
  Threshold t{0, 0.5f};
  auto point = make_point(0.99f);
  auto result = Intersection::intersect_point_with_threshold(t, point, config);
  REQUIRE_FALSE(result.flows_to_left_subtree);
  REQUIRE(result.flows_to_right_subtree);
}

// ── multi-feature ─────────────────────────────────────────────────────────

static Configuration make_config_2f(float left0, float right0, float left1,
                                    float right1) {
  Configuration config;
  config.number_of_features = 2;
  config.perturbations_per_feature = {{left0, right0}, {left1, right1}};
  config.epsilons_per_feature = {{0.0f, 0.0f}, {0.0f, 0.0f}};
  return config;
}

TEST_CASE("Intersection: split on feature 1, point goes right via feature 1",
          "[intersection]") {
  // feature 0 = 0.1 (would go left), feature 1 = 0.8 (goes right)
  // split is on feature 1 → only feature 1 matters
  auto config = make_config_2f(0.0f, 0.0f, 0.0f, 0.0f);
  Threshold t{1, 0.5f};
  Datapoint point{{0.1f, 0.8f}, 0, false};
  auto result = Intersection::intersect_point_with_threshold(t, point, config);
  REQUIRE_FALSE(result.flows_to_left_subtree);
  REQUIRE(result.flows_to_right_subtree);
}

TEST_CASE(
    "Intersection: split on feature 1, perturbation applies to feature 1 only",
    "[intersection]") {
  // feature 1 = 0.3, delta_R[1] = 0.3 → 0.6 >= 0.5 → right
  // feature 1 = 0.3, delta_L[1] = 0.0 → 0.3 < 0.5 → left
  // feature 0 has large perturbation that must be ignored
  auto config = make_config_2f(1.0f, 1.0f, 0.0f, 0.3f);
  Threshold t{1, 0.5f};
  Datapoint point{{0.5f, 0.3f}, 0, false};
  auto result = Intersection::intersect_point_with_threshold(t, point, config);
  REQUIRE(result.flows_to_left_subtree);
  REQUIRE(result.flows_to_right_subtree);
}

// ── asymmetric perturbation ───────────────────────────────────────────────

TEST_CASE("Intersection: right perturbation pushes into right, no left push",
          "[intersection]") {
  // x=0.3, delta_R=0.3 → 0.6 >= 0.5 → right; delta_L=0.0 → 0.3 < 0.5 → left
  auto config = make_config(0.0f, 0.3f);
  Threshold t{0, 0.5f};
  Datapoint point{{0.3f}, 0, false};
  auto result = Intersection::intersect_point_with_threshold(t, point, config);
  REQUIRE(result.flows_to_left_subtree);
  REQUIRE(result.flows_to_right_subtree);
}

TEST_CASE("Intersection: left perturbation on point above threshold, still "
          "right only",
          "[intersection]") {
  // x=0.8, delta_L=0.2 → 0.6 >= 0.5 → NOT left (0.6 < 0.5 is false)
  // x=0.8, delta_R=0.0 → 0.8 >= 0.5 → right
  auto config = make_config(0.2f, 0.0f);
  Threshold t{0, 0.5f};
  Datapoint point{{0.8f}, 0, false};
  auto result = Intersection::intersect_point_with_threshold(t, point, config);
  REQUIRE_FALSE(result.flows_to_left_subtree);
  REQUIRE(result.flows_to_right_subtree);
}

TEST_CASE("Intersection: left perturbation pulls point from right into both",
          "[intersection]") {
  // x=0.7, delta_L=0.3 → 0.4 < 0.5 → left; delta_R=0.0 → 0.7 >= 0.5 → right
  auto config = make_config(0.3f, 0.0f);
  Threshold t{0, 0.5f};
  Datapoint point{{0.7f}, 0, false};
  auto result = Intersection::intersect_point_with_threshold(t, point, config);
  REQUIRE(result.flows_to_left_subtree);
  REQUIRE(result.flows_to_right_subtree);
}

// ── boundary conditions ───────────────────────────────────────────────────

TEST_CASE(
    "Intersection: right perturbation exactly reaches threshold → flows right",
    "[intersection]") {
  // x=0.3, delta_R=0.2 → 0.3+0.2=0.5 >= 0.5 → right
  auto config = make_config(0.0f, 0.21f);
  Threshold t{0, 0.5f};
  Datapoint point{{0.3f}, 0, false};
  auto result = Intersection::intersect_point_with_threshold(t, point, config);
  REQUIRE(result.flows_to_right_subtree);
}

TEST_CASE("Intersection: right perturbation just misses threshold → left only",
          "[intersection]") {
  // x=0.3, delta_R=0.19 → 0.49 < 0.5 → NOT right; 0.3 < 0.5 → left
  auto config = make_config(0.0f, 0.19f);
  Threshold t{0, 0.5f};
  Datapoint point{{0.3f}, 0, false};
  auto result = Intersection::intersect_point_with_threshold(t, point, config);
  REQUIRE(result.flows_to_left_subtree);
  REQUIRE_FALSE(result.flows_to_right_subtree);
}

TEST_CASE(
    "Intersection: left perturbation exactly at threshold → does flow left",
    "[intersection]") {
  // x=0.7, delta_L=0.2 → 0.7-0.2=0.5, check: 0.5 < 0.5? NO → not left
  // x=0.7, delta_R=0.0 → 0.7 >= 0.5 → right
  auto config = make_config(0.2f, 0.0f);
  Threshold t{0, 0.5f};
  Datapoint point{{0.7f}, 0, false};
  auto result = Intersection::intersect_point_with_threshold(t, point, config);
  REQUIRE(result.flows_to_left_subtree);
  REQUIRE(result.flows_to_right_subtree);
}
