#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <fstream>
#include <vector>

struct Options 
{
  const std::string file_path = "/home/hmarefat/scratch/torchFOAM/datasetGen/dataset.npy";
  const int64_t in_s          = 9;
  const int64_t hd_s          = 256;
  const int64_t ot_s          = 1;
  const int64_t numbOfEpochs  = 100;
  const int64_t batchSize     = 524288 * 4;
  int64_t batches_per_epoch   = 0;
  torch::DeviceType device    = torch::kCPU;
  const bool printBatches     = false;
  const bool shuffleDataset   = false;
  const bool debug            = true;
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

  if(options.shuffleDataset)
  {
    std::random_shuffle(_dataset.begin(), _dataset.end());
  }
  return _dataset;
}

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset>
{
  DatasetType mdataset;

public:
  CustomDataset(const DatasetType& dataset) : mdataset(dataset) {}

  torch::data::Example<> get(size_t index) override
  {
    auto features = torch::from_blob(mdataset[index].first.data(), {options.in_s}, torch::kFloat64);
    auto labels = torch::from_blob(&mdataset[index].second, {options.ot_s}, torch::kFloat64);

    return {features.clone(), labels.clone()};
  }

  torch::optional<size_t> size() const
  {
    return mdataset.size();
  };
};

class simpleNNImpl : public torch::nn::Module 
{
  torch::nn::Linear lin1, lin2, lin3, lin4, lin5, lin6;
  
public:
  simpleNNImpl(const int64_t in_size, const int64_t hid_size, const int64_t out_size)
    : lin1(torch::nn::LinearOptions(in_size , hid_size)),
      lin2(torch::nn::LinearOptions(hid_size, hid_size/2)),
      lin3(torch::nn::LinearOptions(hid_size/2, hid_size/4)),
      lin4(torch::nn::LinearOptions(hid_size/4, hid_size/8)),
      lin5(torch::nn::LinearOptions(hid_size/8, hid_size/8)),
      lin6(torch::nn::LinearOptions(hid_size/8, out_size)) 
  {
    register_module("linear1", lin1);
    register_module("linear2", lin2);
    register_module("linear3", lin3);
    register_module("linear4", lin4);
    register_module("linear5", lin5);
    register_module("linear6", lin6);
  }

  torch::Tensor forward(torch::Tensor& x) 
  {
    x = torch::relu(lin1->forward(x));
    x = torch::dropout(x, 0.2, is_training());
    x = torch::relu(lin2->forward(x));
    x = torch::dropout(x, 0.2, is_training());
    x = torch::relu(lin3->forward(x)); 
    x = torch::dropout(x, 0.2, is_training());
    x = torch::relu(lin4->forward(x)); 
    x = torch::dropout(x, 0.2, is_training());
    x = torch::relu(lin5->forward(x));  
    x = torch::dropout(x, 0.2, is_training());
    x = lin6->forward(x);
    return x;
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
  std::cout << "Available device: "
            << (options.device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

  auto raw_dataset  = datasetCreator();
  auto dataset      = CustomDataset(raw_dataset).map(torch::data::transforms::Stack<>());
  auto dataset_size = dataset.size().value();
  options.batches_per_epoch = dataset_size / options.batchSize;

  std::cout << "+-- shuffle     : " << options.shuffleDataset    << "\n";
  std::cout << "+-- dataset size: " << dataset_size              << "\n";
  std::cout << "+-- batch size  : " << options.batchSize         << "\n";
  std::cout << "+-- epochs #    : " << options.numbOfEpochs      << "\n";
  std::cout << "+-- batch/epoch : " << options.batches_per_epoch << "\n";

  auto data_loader  = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>
                    ( std::move(dataset), options.batchSize);

  if (options.printBatches)
  {
    int count = 0;
    for (torch::data::Example<>& batch : *data_loader) 
    {
      std::cout << count++ << " " << "Batch size: " << batch.data.size(0) << " | Labels: ";
      
      for (int64_t i = 0; i < batch.data.size(0); i++) 
      {
        std::cout << batch.target[i].item<double>() << " ";
      }
      std::cout << std::endl;
    }
  }
  
  simpleNN model(options.in_s, options.hd_s, options.ot_s);
  model->to(options.device);
  printf("+-- model is loaded to the device and trainig is starting!\n");

  torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));

  for(int64_t epoch=0; epoch <= options.numbOfEpochs; epoch++)
  {
    int64_t batch_index = 0;
    for(torch::data::Example<>& batch : *data_loader)
    {
      //model->zero_grad();
      auto feat = batch.data.to(options.device).to(torch::kFloat32);
      std::cout << feat[0] << "\n\n";
      auto labs = batch.target.to(options.device).to(torch::kFloat32);
      std::cout << labs[0] << "\n\n";
      auto pred = model->forward(feat);
      std::cout << pred[0] << "\n\n";
      auto loss = torch::nn::functional::mse_loss(pred, labs);

      break;

      optimizer.zero_grad();
      loss.backward();
      optimizer.step();

      std::printf("\r[%2ld/%2ld][%3ld/%3ld] loss: %.4f",
                  epoch,
                  options.numbOfEpochs,
                  batch_index++,
                  options.batches_per_epoch,
                  loss.item<float>());
    }
    break;
  }

  return 0;
}
