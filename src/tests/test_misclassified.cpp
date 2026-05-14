#include <catch2/catch_test_macros.hpp>
#include "build_tree.h"
#include "misclassified_datapoints.h"

TEST_CASE("MisclassifiedEntries: empty → count = 0", "[misclassified]") {
    MisclassifiedEntries m;
    REQUIRE(m.get_number_of_misclassified() == 0);
}

TEST_CASE("MisclassifiedEntries: set one misclassified → count = 1", "[misclassified]") {
    MisclassifiedEntries m;
    auto node = TreeBuilder::build_tree(0, 5);
    m.set_misclassified(0, true, node);
    REQUIRE(m.get_number_of_misclassified() == 1);
}

TEST_CASE("MisclassifiedEntries: set same point true again → count stays 1 (idempotent)", "[misclassified]") {
    MisclassifiedEntries m;
    auto node = TreeBuilder::build_tree(0, 5);
    m.set_misclassified(0, true, node);
    m.set_misclassified(0, true, node);
    REQUIRE(m.get_number_of_misclassified() == 1);
}

TEST_CASE("MisclassifiedEntries: same point re-set at different node → first node wins, count stays 1", "[misclassified]") {
    MisclassifiedEntries m;
    auto node_a = TreeBuilder::build_tree(0, 3);
    auto node_b = TreeBuilder::build_tree(0, 7);
    m.set_misclassified(42, true, node_a);
    m.set_misclassified(42, true, node_b);
    REQUIRE(m.get_number_of_misclassified() == 1);
    REQUIRE(m.is_datapoint_misclassified_at_node(42, 3));
    REQUIRE_FALSE(m.is_datapoint_misclassified_at_node(42, 7));
}

TEST_CASE("MisclassifiedEntries: multiple distinct points → correct count", "[misclassified]") {
    MisclassifiedEntries m;
    auto node = TreeBuilder::build_tree(0, 0);
    m.set_misclassified(0, true, node);
    m.set_misclassified(1, true, node);
    m.set_misclassified(2, true, node);
    REQUIRE(m.get_number_of_misclassified() == 3);
}

TEST_CASE("MisclassifiedEntries: un-misclassify → count decreases", "[misclassified]") {
    MisclassifiedEntries m;
    auto node = TreeBuilder::build_tree(0, 0);
    m.set_misclassified(0, true, node);
    m.set_misclassified(1, true, node);
    m.set_misclassified(0, false, node);
    REQUIRE(m.get_number_of_misclassified() == 1);
}

TEST_CASE("MisclassifiedEntries: un-misclassify all → count = 0", "[misclassified]") {
    MisclassifiedEntries m;
    auto node = TreeBuilder::build_tree(0, 0);
    m.set_misclassified(10, true, node);
    m.set_misclassified(10, false, node);
    REQUIRE(m.get_number_of_misclassified() == 0);
}

TEST_CASE("MisclassifiedEntries: is_misclassified_at_node → true for correct node", "[misclassified]") {
    MisclassifiedEntries m;
    auto node = TreeBuilder::build_tree(0, 7);
    m.set_misclassified(42, true, node);
    REQUIRE(m.is_datapoint_misclassified_at_node(42, 7));
}

TEST_CASE("MisclassifiedEntries: is_misclassified_at_node → false for wrong node", "[misclassified]") {
    MisclassifiedEntries m;
    auto node = TreeBuilder::build_tree(0, 7);
    m.set_misclassified(42, true, node);
    REQUIRE_FALSE(m.is_datapoint_misclassified_at_node(42, 99));
}

TEST_CASE("MisclassifiedEntries: is_misclassified_at_node → false for non-misclassified point", "[misclassified]") {
    MisclassifiedEntries m;
    REQUIRE_FALSE(m.is_datapoint_misclassified_at_node(0, 0));
}

TEST_CASE("MisclassifiedEntries: different points at different nodes → independent", "[misclassified]") {
    MisclassifiedEntries m;
    auto node_a = TreeBuilder::build_tree(0, 3);
    auto node_b = TreeBuilder::build_tree(0, 4);
    m.set_misclassified(10, true, node_a);
    m.set_misclassified(20, true, node_b);
    REQUIRE(m.is_datapoint_misclassified_at_node(10, 3));
    REQUIRE(m.is_datapoint_misclassified_at_node(20, 4));
    REQUIRE_FALSE(m.is_datapoint_misclassified_at_node(10, 4));
    REQUIRE_FALSE(m.is_datapoint_misclassified_at_node(20, 3));
    REQUIRE(m.get_number_of_misclassified() == 2);
}

TEST_CASE("MisclassifiedEntries: after un-misclassify, is_misclassified returns false", "[misclassified]") {
    MisclassifiedEntries m;
    auto node = TreeBuilder::build_tree(0, 0);
    m.set_misclassified(5, true, node);
    REQUIRE(m.is_datapoint_misclassified_at_node(5, 0));
    m.set_misclassified(5, false, node);
    REQUIRE_FALSE(m.is_datapoint_misclassified_at_node(5, 0));
    REQUIRE(m.get_number_of_misclassified() == 0);
}
