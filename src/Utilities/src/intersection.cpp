#include "intersection.h"
#include "datapoint.h"
#include "perturbation.h"
FlowToLeftAndRightSubtree Intersection::intersect_point_with_threshold(
    Threshold &threshold, Datapoint &datapoint, Configuration &configuration) {
  FeatureId feature = threshold.featureId;
  PerturbationAmount perturbation_to_the_right =
      configuration.perturbations_per_feature[feature].right_perturbation;
  PerturbationAmount perturbation_to_the_left =
      configuration.perturbations_per_feature[feature].left_perturbation;
  bool datapoint_can_flow_to_the_right =
      datapoint.feature_values[feature] + perturbation_to_the_right >
      threshold.threshold_value;
  bool datapoint_can_flow_to_the_left =
      datapoint.feature_values[feature] - perturbation_to_the_left <=
      threshold.threshold_value;
  return FlowToLeftAndRightSubtree(datapoint_can_flow_to_the_left,
                                   datapoint_can_flow_to_the_right);
}
