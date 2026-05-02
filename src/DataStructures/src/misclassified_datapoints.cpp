#include "misclassified_datapoints.h"
#include "datapoint.h"
#include <cassert>
Loss MisclassifiedEntries::get_number_of_misclassified() {
  return static_cast<Loss>(this->entry_to_misclassifying_node.size());
}
bool MisclassifiedEntries::set_misclassified(DatapointId datapoint,
                                             bool misclassified, NodePtr tree) {
  assert(tree != nullptr);
  assert(datapoint >= 0);
  const auto &entry = this->entry_to_misclassifying_node.find(datapoint);
  if (entry != this->entry_to_misclassifying_node.end()) {
    return entry->second->node_id == tree->node_id;
  }
  this->entry_to_misclassifying_node.insert({datapoint, tree});
  return true;
}
