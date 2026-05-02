#ifndef INTERSECTION
#define INTERSECTION
#include "configuration.h"
#include "node.h"
struct FlowToLeftAndRightSubtree {
  bool flows_to_left_subtree, flows_to_right_subtree;
  FlowToLeftAndRightSubtree(bool flows_to_left_subtree,
                            bool flows_to_right_subtree)
      : flows_to_left_subtree(flows_to_left_subtree),
        flows_to_right_subtree(flows_to_right_subtree) {}
};
class Intersection {
public:
  static FlowToLeftAndRightSubtree
  intersect_point_with_threshold(Threshold &threshold, Datapoint &datapoint,
                                 Configuration &configuration);
};
#endif
