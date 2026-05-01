#ifndef SIMPLE_NODE_SCHEDULER
#define SIMPLE_NODE_SCHEDULER
#include "node.h"
#include "node_expansion_scheduler_abstract_class.h"
class SimpleScheduler : public Scheduler {
  SimpleScheduler(NodePtr tree);
  bool all_nodes_expanded();
  NodePtr get_next_node();
  void set_expanded(NodeId node_id, bool expansion_status);
};
#endif
