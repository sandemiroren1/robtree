#ifndef PERTURBATION
#define PERTURBATION
#include "datapoint.h"
using PerturbationAmount = FeatureValue;
struct PerturbationOfFeature {
  PerturbationAmount left_perturbation, right_perturbation;
};
using PerturbationsPerFeature = std::vector<PerturbationOfFeature>;
#endif
