#ifndef SIMPLE_NODE_SCHEDULER_FWD
#define SIMPLE_NODE_SCHEDULER_FWD

#include "node.fwd.h"
#include <set>
#include <vector>
using NodePtrList = std::vector<NodePtr>;
using ExpandedNodes = std::set<NodeId>;
class SimpleScheduler;
#endif
