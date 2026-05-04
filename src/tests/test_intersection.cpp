#include <catch2/catch_test_macros.hpp>
#include "intersection.h"

static Configuration make_config(float left_pert, float right_pert) {
    Configuration config;
    config.number_of_features = 1;
    config.perturbations_per_feature = {{left_pert, right_pert}};
    config.epsilons_per_feature      = {{0.0f, 0.0f}};
    return config;
}

static Datapoint make_point(float x) {
    return {{x}, 0, false};
}

TEST_CASE("Intersection: point below threshold, no perturbation → left only", "[intersection]") {
    auto config = make_config(0.0f, 0.0f);
    Threshold t{0, 0.5f};
    auto point = make_point(0.2f);
    auto result = Intersection::intersect_point_with_threshold(t, point, config);
    REQUIRE(result.flows_to_left_subtree);
    REQUIRE_FALSE(result.flows_to_right_subtree);
}

TEST_CASE("Intersection: point above threshold, no perturbation → right only", "[intersection]") {
    auto config = make_config(0.0f, 0.0f);
    Threshold t{0, 0.5f};
    auto point = make_point(0.8f);
    auto result = Intersection::intersect_point_with_threshold(t, point, config);
    REQUIRE_FALSE(result.flows_to_left_subtree);
    REQUIRE(result.flows_to_right_subtree);
}

TEST_CASE("Intersection: point exactly at threshold, no perturbation → both", "[intersection]") {
    auto config = make_config(0.0f, 0.0f);
    Threshold t{0, 0.5f};
    auto point = make_point(0.5f);
    auto result = Intersection::intersect_point_with_threshold(t, point, config);
    REQUIRE(result.flows_to_left_subtree);
    REQUIRE(result.flows_to_right_subtree);
}

TEST_CASE("Intersection: perturbation pushes point into both subtrees", "[intersection]") {
    // x=0.45, delta_R=0.1 → 0.45+0.1=0.55 >= 0.5 (right)
    // x=0.45, delta_L=0.1 → 0.45-0.1=0.35 <= 0.5 (left)
    auto config = make_config(0.1f, 0.1f);
    Threshold t{0, 0.5f};
    auto point = make_point(0.45f);
    auto result = Intersection::intersect_point_with_threshold(t, point, config);
    REQUIRE(result.flows_to_left_subtree);
    REQUIRE(result.flows_to_right_subtree);
}

TEST_CASE("Intersection: large perturbation, point far left still reaches right", "[intersection]") {
    auto config = make_config(0.0f, 1.0f);
    Threshold t{0, 0.5f};
    auto point = make_point(0.0f);  // 0.0 + 1.0 >= 0.5 → right; 0.0 - 0.0 <= 0.5 → left
    auto result = Intersection::intersect_point_with_threshold(t, point, config);
    REQUIRE(result.flows_to_left_subtree);
    REQUIRE(result.flows_to_right_subtree);
}

TEST_CASE("Intersection: point far above threshold with no perturbation → right only", "[intersection]") {
    auto config = make_config(0.0f, 0.0f);
    Threshold t{0, 0.5f};
    auto point = make_point(0.99f);
    auto result = Intersection::intersect_point_with_threshold(t, point, config);
    REQUIRE_FALSE(result.flows_to_left_subtree);
    REQUIRE(result.flows_to_right_subtree);
}
