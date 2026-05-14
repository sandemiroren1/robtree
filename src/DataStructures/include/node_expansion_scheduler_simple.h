#ifndef SIMPLE_NODE_SCHEDULER
#define SIMPLE_NODE_SCHEDULER
#include "node_expansion_scheduler_abstract_class.h"
#include <set>
using NodePtrList = std::vector<NodePtr>;
using ExpandedNodes = std::set<NodeId>;
class SimpleScheduler : public Scheduler {
public:
  SimpleScheduler(NodePtr tree);
  bool all_nodes_expanded() const;
  NodePtr get_next_node() const;
  void set_expanded(NodeId node_id, bool expansion_status);
  bool get_expanded(NodeId node_id) const;

private:
  NodePtrList schedule;
  ExpandedNodes expanded_nodes;
  NodePtrList::size_type index_of_next_node_in_schedule;
  void dfs(NodePtr tree, NodePtrList &schedule);
};
#endif
