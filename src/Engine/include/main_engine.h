#ifndef MAIN_ENGINE
#define MAIN_ENGINE
#include "common.h"
#include "configuration.h"
#include "misclassified_datapoints.h"
#include "node_expansion_scheduler_abstract_class.h"
#include "problem.h"
class Solver {
public:
  Configuration configuration;
  Solver(Configuration configuration);
  Loss solve(Datapoints &datapoints, Depth d);

private:
  Loss solve(MisclassifiedEntries &misclassified_datapoints, NodePtr tree,
             Scheduler &node_expansion_scheduler_abstract_class);
  Loss solve_leaf_node(MisclassifiedEntries &misclassified_datapoints,
                       NodePtr &tree, Scheduler &node_expansion_schedule);
  Loss solve_node(MisclassifiedEntries &misclassified_datapoints, NodePtr &tree,
                  Scheduler &node_expansion_schedule, FeatureId featureId);
};
#endif
