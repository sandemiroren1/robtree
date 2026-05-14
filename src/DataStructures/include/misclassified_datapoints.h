#ifndef MISCLASSIFIED
#define MISCLASSIFIED
#include "common.h"
#include "datapoint.h"
#include "node.h"
#include <map>
#include <sys/types.h>
using EntryNodePair = std::pair<DatapointId, NodePtr>;
using DataPointToMisclassifyingNode = std::map<DatapointId, NodePtr>;
struct MisclassifiedEntries {
  DataPointToMisclassifyingNode entry_to_misclassifying_node;
  void set_misclassified(DatapointId datapoint_id, bool misclassified,
                         NodePtr node);
  bool is_datapoint_misclassified_at_node(DatapointId datapoint_id,
                                          NodeId node_id);
  Loss get_number_of_misclassified();
};
#endif
