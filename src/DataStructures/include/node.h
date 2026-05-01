#ifndef NODE
#define NODE
#include <memory>
using Classification = bool;
using NodeId = char;
struct Node {
  std::shared_ptr<Node> left, right;
  Classification classification;
  NodeId node_id;
  bool is_leaf() const { return left == nullptr && right == nullptr; }
};
using NodePtr = std::shared_ptr<Node>;
#endif
