#ifndef NODE_SCHEDULER
#define NODE_SCHEDULER
#include "node.h"
class Scheduler {

public:
  Scheduler(NodePtr) {}
  virtual bool all_nodes_expanded() const = 0;
  virtual NodePtr get_next_node() const = 0;
  virtual void set_expanded(NodeId node_id, bool expansion_status) = 0;
  virtual bool get_expanded(NodeId node_id) const = 0;
  virtual ~Scheduler() = default;
};
#endif
