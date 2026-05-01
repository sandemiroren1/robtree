#ifndef MAIN_ENGINE
#define MAIN_ENGINE
#include "misclassified_datapoints.h"

#include "node_expansion_scheduler_abstract_class.h"
#include "problem.h"
using Loss = int;
class Solver {
public:
  Loss solve(Problem problem, MisclassifiedEntries misclassified_datapoints,
             NodePtr tree, Scheduler node_expansion_scheduler_abstract_class);

private:
  Loss solve_leaf_node(Problem problem,
                       MisclassifiedEntries misclassified_datapoints,
                       NodePtr tree, Scheduler node_expansion_schedule);
};
#endif
