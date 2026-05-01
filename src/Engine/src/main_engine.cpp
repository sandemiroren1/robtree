#include "main_engine.h"
#include "misclassified_datapoints.h"
#include "node.h"
Loss Solver::solve(Problem problem,
                   MisclassifiedEntries misclassified_datapoints, NodePtr tree,
                   Scheduler node_expansion_schedule) {
  if (tree->is_leaf()) {
    return solve_leaf_node(problem, misclassified_datapoints, tree,
                           node_expansion_schedule);
  }
  return 0;
}

Loss solve_leaf_node(Problem problem,
                     MisclassifiedEntries misclassified_datapoints,
                     NodePtr tree, Scheduler node_expansion_schedule) {
  return 0;
}
