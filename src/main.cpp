#include "main_engine.h"
#include <iostream>

static Configuration make_config(size_t n_features, float perturbation = 0.0f) {
  Configuration config;
  config.number_of_features = n_features;
  config.perturbations_per_feature.assign(n_features,
                                          {perturbation, perturbation});
  config.epsilons_per_feature.assign(n_features, {0.0f, 0.0f});
  return config;
}
int main() {

  Datapoints datapoints = {
      // {{0.5f}, 0, false},
      // {{0.5f}, 1, true},
  };
  Solver solver(make_config(2));
  Loss res = solver.solve(datapoints, 2);
  if (res == 0) {
    std::cout << "yippe";
  }
  printf("%d\n", res);
  return 0;
}
