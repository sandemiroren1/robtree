#ifndef PROBLEM
#define PROBLEM
#include "datapoint.h"
using Depth = int;
struct Problem {
  Datapoints datapoints;
  Depth maximum_depth_allowed;
};
#endif
