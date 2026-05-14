#include "main_engine.h"
#include <iostream>

static Configuration make_config(size_t n_features, float perturbation = 0.0f) {
  Configuration config;
  config.number_of_features = n_features;
  config.perturbations_per_feature.assign(n_features,
                                          {perturbation, perturbation});
  config.epsilons_per_feature.assign(n_features, {0.00001f, 0.00001f});
  return config;
}
int main() {

  Datapoints datapoints = {
      {{0.05f}, 0, false}, {{0.15f}, 1, false}, {{0.25f}, 2, false},
      {{0.35f}, 3, false}, {{0.45f}, 4, false}, {{0.55f}, 5, true},
      {{0.65f}, 6, true},  {{0.75f}, 7, true},  {{0.85f}, 8, true},
      {{0.95f}, 9, true},
  };
  Solver solver(make_config(1, 0.04f));
  Loss res = solver.solve(datapoints, 1);
  printf("%d\n", res);
  return 0;
}
