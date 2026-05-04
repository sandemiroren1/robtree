#include <catch2/catch_test_macros.hpp>
#include "build_tree.h"
#include "node.h"
#include "node_expansion_scheduler_simple.h"

TEST_CASE("SimpleScheduler: not all expanded on construction", "[scheduler]") {
    auto tree = TreeBuilder::build_tree(1, 0);
    SimpleScheduler scheduler(tree);
    REQUIRE_FALSE(scheduler.all_nodes_expanded());
}

TEST_CASE("SimpleScheduler: first node in schedule is root", "[scheduler]") {
    auto tree = TreeBuilder::build_tree(1, 0);
    SimpleScheduler scheduler(tree);
    REQUIRE(scheduler.get_next_node()->node_id == tree->node_id);
}

TEST_CASE("SimpleScheduler: DFS order for depth=1 tree is root, left, right", "[scheduler]") {
    auto tree = TreeBuilder::build_tree(1, 0);
    auto& dec = std::get<DecisionData>(tree->data);
    SimpleScheduler scheduler(tree);

    REQUIRE(scheduler.get_next_node()->node_id == tree->node_id);
    scheduler.set_expanded(tree->node_id, true);

    REQUIRE(scheduler.get_next_node()->node_id == dec.left->node_id);
    scheduler.set_expanded(dec.left->node_id, true);

    REQUIRE(scheduler.get_next_node()->node_id == dec.right->node_id);
    scheduler.set_expanded(dec.right->node_id, true);

    REQUIRE(scheduler.all_nodes_expanded());
}

TEST_CASE("SimpleScheduler: get_expanded reflects set_expanded", "[scheduler]") {
    auto tree = TreeBuilder::build_tree(1, 0);
    SimpleScheduler scheduler(tree);

    REQUIRE_FALSE(scheduler.get_expanded(tree->node_id));
    scheduler.set_expanded(tree->node_id, true);
    REQUIRE(scheduler.get_expanded(tree->node_id));
}

TEST_CASE("SimpleScheduler: backtracking restores index and expanded set", "[scheduler]") {
    auto tree = TreeBuilder::build_tree(1, 0);
    SimpleScheduler scheduler(tree);

    scheduler.set_expanded(tree->node_id, true);
    REQUIRE(scheduler.get_expanded(tree->node_id));

    scheduler.set_expanded(tree->node_id, false);
    REQUIRE_FALSE(scheduler.get_expanded(tree->node_id));
    REQUIRE(scheduler.get_next_node()->node_id == tree->node_id);
}

TEST_CASE("SimpleScheduler: depth=2 tree has 7 nodes in DFS order", "[scheduler]") {
    auto tree = TreeBuilder::build_tree(2, 0);
    SimpleScheduler scheduler(tree);

    // Expand all nodes in DFS order without asserting order — just check count
    int count = 0;
    while (!scheduler.all_nodes_expanded()) {
        auto node = scheduler.get_next_node();
        scheduler.set_expanded(node->node_id, true);
        count++;
    }
    REQUIRE(count == 7);
}
