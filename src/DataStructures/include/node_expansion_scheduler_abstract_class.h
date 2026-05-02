#ifndef NODE_SCHEDULER
#define NODE_SCHEDULER
#include "node.h"
class Scheduler {
  Scheduler(NodePtr tree);

public:
  virtual bool all_nodes_expanded();
  virtual NodePtr get_next_node();
  virtual void set_expanded(NodeId node_id, bool expansion_status);
  virtual bool get_expanded(NodeId node_id);
};
#endif
