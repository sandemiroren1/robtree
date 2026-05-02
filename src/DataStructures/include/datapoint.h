#ifndef DATAPOINT
#define DATAPOINT
#include <memory>
#include <sys/types.h>
#include <vector>
using FeatureValue = float;
using DatapointId = u_int32_t;
using FeatureValues = std::vector<FeatureValue>;
using Classification = bool;
using FeatureId = u_int32_t;
struct Datapoint {
  FeatureValues feature_values;
  DatapointId datapoint_id;
  Classification classification;
};

using Datapoints = std::vector<Datapoint>;
using DatapointPtr = std::shared_ptr<Datapoint>;
#endif
