#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <fstream>
#include <vector>

struct Options 
{
  const std::string file_path = "/home/hmarefat/scratch/torchFOAM/datasetGen/dataset.npy";
  const int64_t in_s = 9;
  const int64_t hd_s = 50;
  const int64_t ot_s = 1;
  const int64_t kBatchSize = 128;
  int64_t batches_per_epoch = 0;
  torch::DeviceType device = torch::kCPU;
};

static Options options;

using DatasetType = std::vector<std::pair<std::vector<double>, double>>;

DatasetType datasetCreator() 
{
  DatasetType _dataset;

  std::ifstream fistream(options.file_path);
  assert(fistream.is_open());

  double u,v,w,s1,s2,s3,s4,s5,s6,cs;
  while(fistream >> u)
  {
    std::vector<double> tmp_inp;
    fistream >> v >> w >> s1 >> s2 
             >> s3 >> s4 >> s5 >> s6
             >> cs; 
    tmp_inp.push_back(u);
    tmp_inp.push_back(v);
    tmp_inp.push_back(w);
    tmp_inp.push_back(s1);
    tmp_inp.push_back(s2);
    tmp_inp.push_back(s3);
    tmp_inp.push_back(s4);
    tmp_inp.push_back(s5);
    tmp_inp.push_back(s6);
    _dataset.push_back(std::make_pair(tmp_inp, cs));
  } 

  std::random_shuffle(_dataset.begin(), _dataset.end());
  
  return _dataset;
}

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset>
{
  DatasetType mdataset;

public:
  CustomDataset(const DatasetType& dataset) : mdataset(dataset) {}

  torch::data::Example<> get(size_t index) 
  {
    auto features = torch::from_blob(&mdataset[index].first, {options.in_s}, torch::kDouble);
    auto labels = torch::from_blob(&mdataset[index].second, {options.ot_s}, torch::kDouble);

    return {features, labels};
  }

  torch::optional<size_t> size() const
  {
    return mdataset.size();
  };
};


class simpleNNImpl : public torch::nn::Module 
{
  torch::nn::Linear lin1, lin2, lin3, lin4;
public:
  simpleNNImpl(const int64_t in_size, const int64_t hid_size, const int64_t out_size)
    : lin1(in_size, hid_size),
      lin2(hid_size, hid_size),
      lin3(hid_size, hid_size),
      lin4(hid_size, out_size) 
  {
    register_module("linear1", lin1);
    register_module("linear2", lin2);
    register_module("linear3", lin3);
    register_module("linear4", lin4);
  }

  torch::Tensor forward(torch::Tensor x) 
  {
    x = lin1->forward(x);
    x = lin2->forward(x);
    x = lin3->forward(x);  
    return torch::sigmoid(lin4->forward(x));
  }

};
TORCH_MODULE(simpleNN);

int main() 
{
  std::cout << "PyTorch version: "
  << TORCH_VERSION_MAJOR << "."
  << TORCH_VERSION_MINOR << "."
  << TORCH_VERSION_PATCH << std::endl;

  if (torch::cuda::is_available())
    options.device = torch::kCUDA;
  std::cout << "Running on: "
            << (options.device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;
/*
  torch::Device device(torch::kCPU);
  std::cout << "CUDA DEVICE COUNT: " << torch::cuda::device_count() << std::endl;
  if (torch::cuda::is_available()) 
  {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::kCUDA;
  }
*/
  //auto options = torch::TensorOptions()
  //      .dtype(torch::kFloat64)
  //      .requires_grad(true);

/*
  std::vector<std::vector<double>> features;
  std::vector<double> labels;
    
  const std::string path = "/home/hmarefat/scratch/torchFOAM/datasetGen/dataset.npy";
  std::ifstream fin(path);

  int count = 0;
  double u,v,w,s1,s2,s3,s4,s5,s6,cs;
  while(fin >> u)
  {
    std::vector<double> tmp_inp;
    fin >> v >> w >> s1 >> s2 
        >> s3 >> s4 >> s5 >> s6
        >> cs;  
    tmp_inp.push_back(u);
    tmp_inp.push_back(v);
    tmp_inp.push_back(w);
    tmp_inp.push_back(s1);
    tmp_inp.push_back(s2);
    tmp_inp.push_back(s3);
    tmp_inp.push_back(s4);
    tmp_inp.push_back(s5);
    tmp_inp.push_back(s6);
    features.push_back(tmp_inp);
    labels.push_back(cs);
    count++;
  }

  std::cout << "+-- features info: " << features.size() << " --- " << features.front().size() << "\n";
  std::cout << "+-- labels info: " << labels.size() << "\n";
  torch::Tensor training_feat = torch::from_blob(features.data(), 
                              { static_cast<unsigned int>(features.size()),
                                static_cast<unsigned int>(features.front().size())});
  torch::Tensor training_labs = torch::from_blob(labels.data(), 
                              { static_cast<unsigned int>(labels.size()), 1});
  std::cout << "+-- Shape of training feature tensor: " << training_feat.sizes() << "\n";
  std::cout << "+-- Shape of training labels tensor: " << training_labs.sizes() << "\n";


  const std::string path = "/home/hmarefat/scratch/torchFOAM/datasetGen/dataset.npy";
  const int64_t in_s = 9;
  const int64_t hd_s = 50;
  const int64_t ot_s = 1;
  const int64_t kBatchSize = 128;

  const int64_t batches_per_epoch = count / options.kBatchSize;
  std::cout << "+-- batch info: " << batches_per_epoch << " " << count << "\n";
*/

  auto raw_dataset = datasetCreator();
  auto dataset = CustomDataset(raw_dataset).map(torch::data::transforms::Stack<>());
  auto dataset_size = dataset.size().value();
  auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>
                   ( std::move(dataset), options.kBatchSize);

  
  std::cout << "data_loader size: " << data_loader.data.size(0) << "\n";
  /*
  int count = 0;
  for (torch::data::Example<>& batch : *data_loader) 
  {
    std::cout << count++ << " " << "Batch size: " << batch.data.size(0); << " | Labels: ";
    
    for (int64_t i = 0; i < batch.data.size(0); ++i) 
    {
      std::cout << batch.target[i].item<double>() << " ";
    }
   
    std::cout << std::endl;
  }
  */
  
  /*
  simpleNN model(options.in_s, options.hd_s, options.ot_s);
  */
  //training_feat.to(device);
  //training_labs.to(device);
  //model->to(device);
  //std::cout << "+-- data and model are loaded to the device!" << "\n";

  //torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(2e-4));

  //torch::Tensor pred = model->forward(training_feat);
  //std::cout << "+-- sample data at the begining of label vec: " << training_labs[0] << "\n" << pred[0] << std::endl;
/*
  for(int64_t epoch=1; epoch <= kNumberOfEpochs; ++epoch)
  {
    int64_t batch_index = 0;
    for(torch::data::Example<>& batch : *data_loader)
    {
      model->zero_grad();
      torch::Tensor real_images = batch.data.to(device);
      torch::Tensor real_labels = torch::empty(batch.data.size(0)).uniform_(0.8, 1.0).to(device);
      torch::Tensor real_output = discriminator->forward(real_images);
      torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
      d_loss_real.backward();
    }
  }
*/
  return 0;
}
