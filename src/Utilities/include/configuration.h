#ifndef CONFIGURATION
#define CONFIGURATION
#include "perturbation.h"
#include <cstddef>
struct Configuration {
  size_t number_of_features;
  // Define attacker capabilities
  PerturbationsPerFeature perturbations_per_feature;
  PerturbationsPerFeature epsilons_per_feature;
};
#endif
