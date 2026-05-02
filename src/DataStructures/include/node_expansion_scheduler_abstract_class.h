#ifndef NODE_SCHEDULER
#define NODE_SCHEDULER
#include "node.h"
class Scheduler {

public:
  Scheduler(NodePtr tree);
  virtual bool all_nodes_expanded() const;
  virtual NodePtr get_next_node() const;
  virtual void set_expanded(NodeId node_id, bool expansion_status);
  virtual bool get_expanded(NodeId node_id) const;
};
#endif
