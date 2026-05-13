#include "node_expansion_scheduler_simple.h"
#include "node.h"
#include "node_expansion_scheduler_abstract_class.h"
#include <cassert>
#include <iostream>
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
  // std::cout << "Expanding: " << node_id << " -> " << expansion_status;
  // if (this->index_of_next_node_in_schedule != this->schedule.size()) {
  //
  //   std::cout << " next node to expand: "
  //             <<
  //             this->schedule[this->index_of_next_node_in_schedule]->node_id;
  // } else {
  //   std::cout << "should prpobably be unexpanding rn...";
  // }
  // std::cout << std::endl;
  if (expansion_status) {
    assert(node_id ==
           this->schedule[this->index_of_next_node_in_schedule]->node_id);
    assert(this->index_of_next_node_in_schedule < this->schedule.size());
    assert(this->expanded_nodes.find(node_id) == this->expanded_nodes.end());
    this->expanded_nodes.insert(node_id);
    this->index_of_next_node_in_schedule++; // This increment has to be done
                                            // last
  } else {
    assert(index_of_next_node_in_schedule != 0); // This will cause a crash
    this->index_of_next_node_in_schedule--; // We do this increment as now the
                                            // index is pointing to the end
    assert(this->expanded_nodes.find(node_id) !=
           this->expanded_nodes.end()); // container contains the entry
    this->expanded_nodes.erase(node_id);
  }
  // std::cout << "Expanded!" << std::endl;
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
