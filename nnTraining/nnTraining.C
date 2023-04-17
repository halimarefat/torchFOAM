#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <fstream>
#include <vector>
#include <fstream>

struct Options 
{
  const std::string file_path = "/home/hmarefat/scratch/torchFOAM/datasetGen/dataset.npy";
  const int64_t in_s          = 9;
  const int64_t hd_s          = 256;
  const int64_t ot_s          = 1;
  const int64_t numbOfEpochs  = 600;
  const int64_t batchSize     = 1028; //524288 / 256; // size of one time-step >> 524288;
  int64_t dataset_size        = 0;
  const float train_val_split = 0.2;
  int64_t train_size          = 0;
  int64_t val_size            = 0;
  int64_t train_batches_per_epoch = 0;
  int64_t val_batches_per_epoch   = 0;
  torch::DeviceType device    = torch::kCPU;
  const bool printBatches     = false;
  const bool shuffleDataset   = true;
  const int64_t numOfShuffle  = 10;
  const bool debug            = true;
};

static Options options;

using DatasetType = std::vector<std::pair<std::vector<double>, double>>;

std::pair<DatasetType, DatasetType> datasetCreator() 
{
  DatasetType _dataset;
  DatasetType train, val;

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

  options.dataset_size = _dataset.size();
  
  if(options.shuffleDataset)
  {
    for(int64_t i=0; i<=options.numOfShuffle; i++)
    {
      std::random_shuffle(_dataset.begin(), _dataset.end());
    }
  }

  int64_t split = (1.0 - options.train_val_split) * options.dataset_size;

  int count = 0;
  for(auto r : _dataset)
  {
    if(count < split)
    {
      train.push_back(r);
    }
    if(count > split)
    {
      val.push_back(r);
    }
    count++;
  }

  options.train_size = train.size();
  options.val_size   = val.size();

  if(options.shuffleDataset)
  {
    for(int64_t i=0; i<=options.numOfShuffle; i++)
    {
      std::random_shuffle(train.begin(), train.end());
      std::random_shuffle(val.begin(), val.end());
    }
  }

  return std::make_pair(train, val);
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

  auto train_val_ds = datasetCreator();
  auto train_ds     = CustomDataset(train_val_ds.first).map(torch::data::transforms::Stack<>());
  auto val_ds       = CustomDataset(train_val_ds.second).map(torch::data::transforms::Stack<>());
  options.train_batches_per_epoch = options.train_size / options.batchSize;
  options.val_batches_per_epoch   = options.val_size   / options.batchSize;

  std::cout << "+-- shuffle      : " << options.shuffleDataset    << "\n";
  std::cout << "+-- dataset size : " << options.dataset_size      << "\n";
  std::cout << "+-- train ds size: " << options.train_size        << "\n";
  std::cout << "+-- val ds size  : " << options.val_size          << "\n";
  std::cout << "+-- batch size   : " << options.batchSize         << "\n";
  std::cout << "+-- t batch/epoch: " << options.train_batches_per_epoch << "\n";
  std::cout << "+-- v batch/epoch: " << options.val_batches_per_epoch << "\n";
  std::cout << "+-- epochs #     : " << options.numbOfEpochs      << "\n";

  auto train_data_loader  = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>
                          ( std::move(train_ds), options.batchSize);
  auto val_data_loader    = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>
                          ( std::move(val_ds), options.batchSize);


  if (options.printBatches)
  {
    int count = 0;
    for (torch::data::Example<>& batch : *train_data_loader) 
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
  
  std::ofstream MyFile("filename.txt");

  // Write to the file
  MyFile << "Files can be tricky, but it is fun enough!";

  // Close the file
  MyFile.close();

  std::ofstream logFile("./_train.log");
  assert(logFile.is_open());
  logFile << "epoch\tLoss_train\tAcc_train\tLoss_val\tAcc_val\n";

  float best_loss = std::numeric_limits<float>::max();

  for(int64_t epoch=0; epoch <= options.numbOfEpochs; epoch++)
  {
    int64_t batch_index = 0;
    float Loss_train= 0, Loss_val= 0;
    float Acc_train= 0 , Acc_val= 0;

    for(torch::data::Example<>& batch : *train_data_loader)
    {
      model->train();
      auto train_feat = batch.data.to(options.device).to(torch::kFloat32);
      auto train_labs = batch.target.to(options.device).to(torch::kFloat32);
      auto train_pred = model->forward(train_feat);
      auto train_loss = torch::nn::functional::mse_loss(train_pred, train_labs);
      assert(!std::isnan(train_loss.template item<float>()));
      auto train_acc = train_pred.argmax(1).eq(train_labs).sum();

      optimizer.zero_grad();
      train_loss.backward();
      optimizer.step();

      Loss_train += train_loss.template item<float>();
      Acc_train  += train_acc.template item<float>();
      std::printf("\rTraing Batch: [%3ld/%3ld]", batch_index++, options.train_batches_per_epoch);
    }

    batch_index = 0;
    for(torch::data::Example<>& batch : *val_data_loader)
    {
      model->eval();
      auto val_feat = batch.data.to(options.device).to(torch::kFloat32);
      auto val_labs = batch.target.to(options.device).to(torch::kFloat32);
      auto val_pred = model->forward(val_feat);
      auto val_loss = torch::nn::functional::mse_loss(val_pred, val_labs);
      assert(!std::isnan(val_loss.template item<float>()));
      auto val_acc = val_pred.argmax(1).eq(val_labs).sum();
      val_loss.backward(); 

      Loss_val += val_loss.template item<float>();
      Acc_val  += val_acc.template item<float>();
      std::printf("\rVal Batch: [%3ld/%3ld]", batch_index++, options.val_batches_per_epoch);
    }
    
    std::printf("\r                                >>>> Epoch: [%2ld/%2ld] --- train_loss: %.4f, train_acc: %.4f | val_loss: %.4f, val_acc: %.4f",
                epoch,
                options.numbOfEpochs,
                Loss_train, 
                Acc_train,
                Loss_val, 
                Acc_val); 
    
    logFile << epoch << "\t" << Loss_train << "\t" << Acc_train << "\t" << Loss_val << "\t" << Acc_val << "\n";
    if(Loss_val < best_loss)
    {
      torch::save(model, "./best_model.pt");
      best_loss = Loss_val;
    }
  }

  logFile.close();

  return 0;
}
