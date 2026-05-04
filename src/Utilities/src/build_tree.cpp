#include "build_tree.h"
#include "datapoint.h"
#include "node.h"
#include <memory>
NodePtr TreeBuilder::build_tree(Depth depth, NodeId node_id) {

  Datapoints datapoints;
  if (depth == 0) {

    Classification classification = false;
    return std::make_shared<Node>(Node(node_id, datapoints, classification));
  }
  Threshold threshold;
  NodePtr left = build_tree(depth - 1, node_id * 2 + 1);
  NodePtr right = build_tree(depth - 1, node_id * 2 + 2);
  return std::make_shared<Node>(
      Node(node_id, datapoints, threshold, left, right));
}
