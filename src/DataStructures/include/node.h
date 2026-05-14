#ifndef NODE
#define NODE
#include "datapoint.h"
#include "node.fwd.h"
#include <iostream>
#include <string>
struct Threshold {
  FeatureId featureId;
  FeatureValue threshold_value;
};
struct LeafData {
  Classification classification;
  LeafData(Classification classification) : classification(classification) {};
};

struct DecisionData {
  Threshold threshold;
  NodePtr left, right;
  DecisionData(Threshold threshold, NodePtr left, NodePtr right)
      : threshold(threshold), left((left)), right((right)) {};
};
struct Node {
  Node(NodeId node_id, Datapoints datapoints, Classification classification)
      : node_id(node_id), datapoints(datapoints),
        data(LeafData(classification)) {}
  Node(NodeId node_id, Datapoints datapoints, Threshold threshold, NodePtr left,
       NodePtr right)
      : node_id(node_id), datapoints(datapoints),
        data(DecisionData(threshold, left, right)) {}
  std::variant<LeafData, DecisionData> data;
  NodeId node_id;
  Datapoints datapoints;
  bool is_leaf() const { return std::holds_alternative<LeafData>(data); }
};

inline void print_node(std::ostream& os, const Node& node,
                       const std::string& prefix = "", bool is_left = false) {
  os << prefix;
  if (!prefix.empty()) os << (is_left ? "|-- " : "\\-- ");

  if (node.is_leaf()) {
    const auto& leaf = std::get<LeafData>(node.data);
    os << "[Leaf " << node.node_id << "] class="
       << (leaf.classification ? "1" : "0")
       << " n=" << node.datapoints.size() << "\n";
  } else {
    const auto& dec = std::get<DecisionData>(node.data);
    os << "[Node " << node.node_id << "] f" << dec.threshold.featureId
       << " <= " << dec.threshold.threshold_value
       << " n=" << node.datapoints.size() << "\n";
    std::string child_prefix =
        prefix + (prefix.empty() ? "" : (is_left ? "|   " : "    "));
    if (dec.left)  print_node(os, *dec.left,  child_prefix, true);
    if (dec.right) print_node(os, *dec.right, child_prefix, false);
  }
}

inline std::ostream& operator<<(std::ostream& os, const Node& node) {
  print_node(os, node);
  return os;
}
#endif
