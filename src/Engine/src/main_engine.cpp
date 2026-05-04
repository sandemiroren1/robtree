#include "main_engine.h"
#include "build_tree.h"
#include "configuration.h"
#include "datapoint.h"
#include "intersection.h"
#include "misclassified_datapoints.h"
#include "node.h"
#include "node_expansion_scheduler_simple.h"
#include <cassert>
#include <cmath>
Solver::Solver(Configuration configuration) {
  this->configuration = configuration;
}
Loss Solver::solve(MisclassifiedEntries &misclassified_datapoints, NodePtr tree,
                   Scheduler &node_expansion_schedule) {
  if (tree->is_leaf()) {
    return solve_leaf_node(misclassified_datapoints, tree,
                           node_expansion_schedule);
  }
  Loss result = INFINITY;
  for (size_t feature_id; feature_id < this->configuration.number_of_features;
       feature_id++) {
    result = std::min(result, solve_node(misclassified_datapoints, tree,
                                         node_expansion_schedule, feature_id));
  }
  return result;
}
Loss Solver::solve(Datapoints &datapoints, Depth depth) {
  auto tree = TreeBuilder::build_tree(depth, 0);
  SimpleScheduler scheduler(tree);
  MisclassifiedEntries misclassified_datapoints;
  return solve(misclassified_datapoints, tree, scheduler);
}
Loss Solver::solve_node(MisclassifiedEntries &misclassified_datapoints,
                        NodePtr &tree, Scheduler &node_expansion_schedule,
                        FeatureId featureId) {
  assert(tree != nullptr);
  assert(featureId >= 0);
  assert(!node_expansion_schedule.all_nodes_expanded());
  assert(!node_expansion_schedule.get_expanded(tree->node_id));

  auto &decisionData = std::get<DecisionData>(tree->data);
  decisionData.threshold.featureId = featureId;
  auto &datapoints = tree->datapoints;
  // std::sort(datapoints.begin(), datapoints.end(),
  //           [&](auto &a, auto &b) { a[featureId] <= b[featureId]; });
  Loss result = INFINITY;

  node_expansion_schedule.set_expanded(tree->node_id, true);
  for (auto &entry : datapoints) {
    decisionData.threshold.threshold_value =
        // Its optimal to place thresholds in positions {x1 + dR, x2 + dR,...}.
        // Check my proof out.
        entry.feature_values[featureId] +
        this->configuration.perturbations_per_feature[featureId]
            .right_perturbation;
    decisionData.left->datapoints.clear();
    decisionData.right->datapoints.clear();
    for (auto &datapoint : datapoints) {
      auto intersection = Intersection::intersect_point_with_threshold(
          decisionData.threshold, datapoint, this->configuration);
      if (intersection.flows_to_left_subtree) {
        decisionData.left->datapoints.push_back(datapoint);
      }

      if (intersection.flows_to_right_subtree) {
        decisionData.right->datapoints.push_back(datapoint);
      }
    }
    auto next_node_to_expand = node_expansion_schedule.get_next_node();
    result =
        std::min(result, solve(misclassified_datapoints, next_node_to_expand,
                               node_expansion_schedule));
  }

  node_expansion_schedule.set_expanded(tree->node_id, false);
  return result;
}

Loss Solver::solve_leaf_node(MisclassifiedEntries &misclassified_datapoints,
                             NodePtr &tree,
                             Scheduler &node_expansion_schedule) {

  assert(tree != nullptr);
  assert(!node_expansion_schedule.all_nodes_expanded());
  assert(!node_expansion_schedule.get_expanded(tree->node_id));
  assert(tree->is_leaf());

  auto &leafdata = std::get<LeafData>(tree->data);
  // PHASE 1, make decision and write down the consequences.
  node_expansion_schedule.set_expanded(tree->node_id, true);
  Loss number_of_newly_misclassified = 0;

  for (auto &entry : tree->datapoints) {
    if (entry.classification != leafdata.classification) {
      bool was_point_newly_misclassified =
          misclassified_datapoints.set_misclassified(entry.datapoint_id, true,
                                                     tree);
      number_of_newly_misclassified += was_point_newly_misclassified;
    }
  }
  if (node_expansion_schedule.all_nodes_expanded()) {
    return misclassified_datapoints.get_number_of_misclassified();
  }
  auto next_node_to_expand = node_expansion_schedule.get_next_node();
  Loss result = solve(misclassified_datapoints, next_node_to_expand,
                      node_expansion_schedule);

  // PHASE 2, undo what was done after backtracking!
  node_expansion_schedule.set_expanded(tree->node_id, false);
  // un-misclassify the points that were misclassified on this node.
  for (auto &entry : tree->datapoints) {
    if (entry.classification != leafdata.classification) {
      bool was_point_newly_misclassified =
          misclassified_datapoints.set_misclassified(entry.datapoint_id, true,
                                                     tree);
      if (was_point_newly_misclassified) {
        misclassified_datapoints.set_misclassified(entry.datapoint_id, false,
                                                   tree);
      }
    }
  }
  return result;
}
