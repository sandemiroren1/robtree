#ifndef NODE
#define NODE
#include "NodeProblem.h"
#include "datapoint.h"
#include <memory>
using NodeId = char;
struct Threshold {
  FeatureId featureId;
  FeatureValue threshold_value;
};
struct Node {
  std::shared_ptr<Node> left, right;
  Classification classification;
  NodeId node_id;
  Threshold threshold;
  NodeProblem node_problem;
  bool is_leaf() const { return left == nullptr && right == nullptr; }
};
using NodePtr = std::shared_ptr<Node>;
#endif
