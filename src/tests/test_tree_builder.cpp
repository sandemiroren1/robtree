#include <catch2/catch_test_macros.hpp>
#include "build_tree.h"
#include "node.h"

TEST_CASE("build_tree depth=0 produces a single leaf", "[tree_builder]") {
    auto tree = TreeBuilder::build_tree(0, 0);
    REQUIRE(tree != nullptr);
    REQUIRE(tree->is_leaf());
    REQUIRE(tree->node_id == 0);
    REQUIRE(tree->datapoints.empty());
}

TEST_CASE("build_tree depth=1 produces decision node with two leaf children", "[tree_builder]") {
    auto tree = TreeBuilder::build_tree(1, 0);
    REQUIRE(tree != nullptr);
    REQUIRE_FALSE(tree->is_leaf());

    auto& dec = std::get<DecisionData>(tree->data);
    REQUIRE(dec.left != nullptr);
    REQUIRE(dec.right != nullptr);
    REQUIRE(dec.left->is_leaf());
    REQUIRE(dec.right->is_leaf());
}

TEST_CASE("build_tree depth=2 has correct two-level structure", "[tree_builder]") {
    auto tree = TreeBuilder::build_tree(2, 0);
    REQUIRE_FALSE(tree->is_leaf());

    auto& root_dec = std::get<DecisionData>(tree->data);
    REQUIRE_FALSE(root_dec.left->is_leaf());
    REQUIRE_FALSE(root_dec.right->is_leaf());

    auto& left_dec  = std::get<DecisionData>(root_dec.left->data);
    auto& right_dec = std::get<DecisionData>(root_dec.right->data);
    REQUIRE(left_dec.left->is_leaf());
    REQUIRE(left_dec.right->is_leaf());
    REQUIRE(right_dec.left->is_leaf());
    REQUIRE(right_dec.right->is_leaf());
}

TEST_CASE("build_tree assigns correct node IDs (BFS numbering)", "[tree_builder]") {
    auto tree = TreeBuilder::build_tree(2, 0);
    REQUIRE(tree->node_id == 0);

    auto& root_dec = std::get<DecisionData>(tree->data);
    REQUIRE(root_dec.left->node_id == 1);
    REQUIRE(root_dec.right->node_id == 2);

    auto& left_dec  = std::get<DecisionData>(root_dec.left->data);
    auto& right_dec = std::get<DecisionData>(root_dec.right->data);
    REQUIRE(left_dec.left->node_id == 3);
    REQUIRE(left_dec.right->node_id == 4);
    REQUIRE(right_dec.left->node_id == 5);
    REQUIRE(right_dec.right->node_id == 6);
}

TEST_CASE("build_tree leaves default to classification=false", "[tree_builder]") {
    auto tree = TreeBuilder::build_tree(0, 0);
    auto& leaf = std::get<LeafData>(tree->data);
    REQUIRE(leaf.classification == false);
}
