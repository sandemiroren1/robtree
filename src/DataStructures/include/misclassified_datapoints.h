#ifndef MISCLASSIFIED
#define MISCLASSIFIED
#include "datapoint.h"
#include "node.h"
#include <map>
using EntryNodePair = std::pair<DatapointId, NodePtr>;
using DataPointToMisclassifyingNode = std::map<DatapointId, NodePtr>;
struct MisclassifiedEntries {
  DataPointToMisclassifyingNode entry_to_misclassifying_node;
  void set_misclassified(DatapointId datapoint_id, bool misclassified);
};
#endif
