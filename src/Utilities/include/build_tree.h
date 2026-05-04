#ifndef TREE_BUILDER
#define TREE_BUILDER

#include "node.fwd.h"
#include "problem.h"
struct TreeBuilder {
  static NodePtr build_tree(Depth depth, NodeId node_id);
};
#endif
