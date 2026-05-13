#ifndef NODE
#define NODE
#include "datapoint.h"
#include "node.fwd.h"
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
#endif
