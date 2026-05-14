#include "misclassified_datapoints.h"
#include "datapoint.h"
#include <cassert>
Loss MisclassifiedEntries::get_number_of_misclassified() {
  return static_cast<Loss>(this->entry_to_misclassifying_node.size());
}
bool MisclassifiedEntries::is_datapoint_misclassified_at_node(
    DatapointId datapoint_id, NodeId node_id) {

  assert(datapoint_id >= 0);
  assert(node_id >= 0);
  const auto &entry = this->entry_to_misclassifying_node.find(datapoint_id);
  if (entry == this->entry_to_misclassifying_node.end()) {
    return false;
  }
  // the point is misclassified, but is it by this node?
  return entry->second->node_id == node_id;
}
// Returns if the function newly set an entry as misclassified
void MisclassifiedEntries::set_misclassified(DatapointId datapoint_id,
                                             bool misclassified, NodePtr tree) {
  assert(tree != nullptr);
  assert(datapoint_id >= 0);
  const auto &entry = this->entry_to_misclassifying_node.find(datapoint_id);
  if (entry == this->entry_to_misclassifying_node.end() &&
      misclassified) { // Newly misclassified
    this->entry_to_misclassifying_node.insert({datapoint_id, tree});
  }
  if (!misclassified) { // if we want to un-misclassify an entry
    this->entry_to_misclassifying_node.erase(entry);
  }
  // we dont need to do anything else if its already misclassified
}
