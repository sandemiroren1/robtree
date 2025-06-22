This folder contains other experimental versions of the search algorithm. The code may be a bit messy as the ideas were very quickly abandoned.

### Solver with Subinterval Pruning

Our proposed version of the algorithm iterates over the indices from left to right. We made an additional version of the algorithm that uses the same iteration done as ConTree.
However, this idea was dropped due to the removal of the O(n) overhead from calculating the misclassifications being a better boost.


### Solver old method

The old iteration order with the O(n) time to compute the misclassifications can also be found here. 