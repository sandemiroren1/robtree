#ifndef NODE_FWD
#define NODE_FWD
#include <memory>
using NodeId = char;
struct Node;
using NodePtr = std::shared_ptr<Node>;
#endif
