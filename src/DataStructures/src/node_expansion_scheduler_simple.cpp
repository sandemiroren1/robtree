#include "node_expansion_scheduler_simple.h"
#include "node.h"
#include "node_expansion_scheduler_abstract_class.h"
#include <cassert>
SimpleScheduler::SimpleScheduler(NodePtr tree) : Scheduler(tree) {

  NodePtrList schedule;
  dfs(tree, schedule);
  this->schedule = schedule;
  this->index_of_next_node_in_schedule = 0;
}
bool SimpleScheduler::all_nodes_expanded() const {
  return this->index_of_next_node_in_schedule >= this->schedule.size();
}
NodePtr SimpleScheduler::get_next_node() const {
  assert(this->index_of_next_node_in_schedule < this->schedule.size());
  auto &node_to_return = this->schedule[this->index_of_next_node_in_schedule];
  assert(node_to_return != nullptr);
  return node_to_return;
}
void SimpleScheduler::set_expanded(NodeId node_id, bool expansion_status) {

  assert(this->index_of_next_node_in_schedule < this->schedule.size());
  assert(node_id ==
         this->schedule[this->index_of_next_node_in_schedule]->node_id);

  if (expansion_status) {
    this->index_of_next_node_in_schedule++;
    assert(this->expanded_nodes.find(node_id) == this->expanded_nodes.end());
    this->expanded_nodes.insert(node_id);
  } else {
    this->index_of_next_node_in_schedule--;
    assert(this->expanded_nodes.find(node_id) != this->expanded_nodes.end());
    this->expanded_nodes.erase(node_id);
  }
}
bool SimpleScheduler::get_expanded(NodeId node_id) const {
  return this->expanded_nodes.find(node_id) != this->expanded_nodes.end();
}
void SimpleScheduler::dfs(NodePtr tree, NodePtrList &schedule) {
  assert(tree != nullptr);
  schedule.push_back(tree);
  if (tree->is_leaf()) {
    return;
  }
  auto &decision = std::get<DecisionData>(tree->data);
  assert(decision.left != nullptr && decision.right != nullptr);
  dfs(decision.left, schedule);
  dfs(decision.right, schedule);
}
