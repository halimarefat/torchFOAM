#include <torch/torch.h>

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset>
{
  std::vector<std::vector<double>> mdataset;
  int64_t m_in_s;

public:
  CustomDataset(std::vector<std::vector<double>>& dataset, int64_t in_s) 
        : mdataset(dataset), 
          m_in_s(in_s)
        {}

  torch::data::Example<> get(size_t index) override
  {
    auto features = torch::from_blob(mdataset[index].data(), {m_in_s}, torch::kFloat64);
    std::vector<double> some_vector{1.0};
    auto labels = torch::from_blob(some_vector.data(), {1}, torch::kFloat64);

    return {features.clone(), labels.clone()};
  }

  torch::optional<size_t> size() const
  {
    return mdataset.size();
  };
};