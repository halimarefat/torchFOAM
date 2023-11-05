#pragma once

#include <torch/torch.h>
#include <iostream>
#include <memory>
#include "config.H"

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset>
{
  DatasetType mdataset;
  Options moptions;

public:
  CustomDataset(const DatasetType& dataset, Options& options) 
        : mdataset(dataset),
          moptions(options) 
         {}

  torch::data::Example<> get(size_t index) override
  {
    auto features = torch::from_blob(mdataset[index].first.data(), {moptions.in_s}, torch::kDouble); //kFloat64);
    auto labels = torch::from_blob(&mdataset[index].second, {moptions.ot_s}, torch::kDouble); //kFloat64);

    return {features.clone(), labels.clone()};
  }

  torch::optional<size_t> size() const
  {
    return mdataset.size();
  };
};
