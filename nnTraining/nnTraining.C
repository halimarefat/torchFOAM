#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <memory>
#include <fstream>
#include <vector>
#include <fstream>
#include "config.H"
#include "utils.H"
#include "CustomDataset.H"
#include "model.H"

static Options options;

int main() 
{
  std::cout << "PyTorch version: "
            << TORCH_VERSION_MAJOR << "."
            << TORCH_VERSION_MINOR << "."
            << TORCH_VERSION_PATCH << std::endl;
  std::cout << "+-- File path    : " << options.file_path << "\n";
  std::cout << "+-- MNum         : " << options.MNum      << "\n";

  if (torch::cuda::is_available())
    options.device = torch::kCUDA;
  std::cout << "Available device: "
            << (options.device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

  auto train_val_ds = datasetCreator(options);
  auto train_ds     = CustomDataset(train_val_ds.first, options).map(torch::data::transforms::Stack<>());
  auto val_ds       = CustomDataset(train_val_ds.second, options).map(torch::data::transforms::Stack<>());
  options.train_batches_per_epoch = options.train_size / options.batchSize;
  options.val_batches_per_epoch   = options.val_size   / options.valBatchSize;

  std::cout << "+-- shuffle      : " << options.shuffleDataset    << "\n";
  std::cout << "+-- dataset size : " << options.dataset_size      << "\n";
  std::cout << "+-- train ds size: " << options.train_size        << "\n";
  std::cout << "+-- batch size   : " << options.batchSize         << "\n";
  std::cout << "+-- val ds size  : " << options.val_size          << "\n";
  std::cout << "+-- ValBatch size: " << options.valBatchSize      << "\n";
  std::cout << "+-- t batch/epoch: " << options.train_batches_per_epoch << "\n";
  std::cout << "+-- v batch/epoch: " << options.val_batches_per_epoch << "\n";
  std::cout << "+-- epochs #     : " << options.numbOfEpochs      << "\n";

  auto train_data_loader  = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>
                          ( std::move(train_ds), options.batchSize);
  auto val_data_loader    = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>
                          ( std::move(val_ds), options.valBatchSize);


  if (options.printBatches)
  {
    int count = 0;
    for (torch::data::Example<>& batch : *train_data_loader) 
    {
      auto features = batch.data;
      auto labels = batch.target;
      std::cout << " Batch No.: " << count++ << " " << "Batch size: " << batch.data.size(0) << "\n";
      std::cout << std::setprecision(15) << " | Feats : " << features[0][0].item<double>() << "\n";
      std::cout << std::setprecision(15) << " | Labels: " << labels[0].item<double>() << "\n";
      /*
      for (int64_t i = 0; i < batch.data.size(0); i++) 
      {
        std::cout << batch.target[i].item<double>() << " ";
      }
      */
      std::cout << std::endl;
      break;
    }
  }

  ANNModel model(options.in_s, options.ot_s, options.hl_n, options.hd_s);
  const std::string bestPATH = "./best_model_" + options.file_name_without_extension +".pt";
  const std::string logPATH = "./log_" + options.file_name_without_extension + ".log";

  if(options.trainMode)
  {
    model->to(options.device);
    model->to(torch::kDouble);
    printf("+-- model is loaded to the device and trainig is starting!\n");

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
    torch::optim::StepLR scheduler(optimizer, /*step_size=*/ 10, /*gamma=*/ 0.5);
    torch::nn::MSELoss loss;

    std::ofstream logFile(logPATH);
    assert(logFile.is_open());
    logFile << "epoch\tLoss_train\tLoss_val\n";

    float best_loss = std::numeric_limits<float>::max();
    int early_stop_counter = 0;

    for(int64_t epoch=1; epoch <= options.numbOfEpochs; epoch++)
    {
      float Loss_train= 0, Loss_val= 0;
      int64_t batch_index = 0;
      
      model->train();
      for(const auto& batch : *train_data_loader)
      {
        auto train_feat = batch.data.to(options.device);
        auto train_labs = batch.target.to(options.device);
        auto train_pred = model->forward(train_feat);
        auto train_loss = loss(train_pred, train_labs);
        assert(!std::isnan(train_loss.template item<float>()));
        //auto train_acc = train_pred.argmax(1).eq(train_labs).sum();

        optimizer.zero_grad();
        train_loss.backward();
        optimizer.step();

        Loss_train += train_loss.template item<float>();
        //Acc_train  += train_acc.template item<float>();
        
        std::printf("\rTraing Batch: [%3ld/%3ld]", batch_index++, options.train_batches_per_epoch);
      }

      Loss_train = Loss_train / options.train_batches_per_epoch;


      batch_index = 0;
      
      model->eval();
      for(const auto& batch : *val_data_loader)
      {
        auto val_feat = batch.data.to(options.device);
        auto val_labs = batch.target.to(options.device);
        auto val_pred = model->forward(val_feat);
        auto val_loss = loss(val_pred, val_labs);
        assert(!std::isnan(val_loss.template item<float>()));
        //auto val_acc = val_pred.argmax(1).eq(val_labs).sum();

        Loss_val += val_loss.template item<float>();
        //Acc_val  += val_acc.template item<float>();
        std::printf("\rVal Batch: [%3ld/%3ld]", batch_index++, options.val_batches_per_epoch);
      }
      Loss_val = Loss_val / options.val_batches_per_epoch;

      std::printf("\r                                >>>> Epoch: [%2ld/%2ld] --- train_loss: %.4f | val_loss: %.4f",
                  epoch,
                  options.numbOfEpochs,
                  Loss_train, 
                  Loss_val); 
      
      logFile << epoch << "\t" << Loss_train << "\t" << Loss_val << "\n";
      logFile.flush();

      scheduler.step();

      if(Loss_val < best_loss)
      {
          torch::save(model, bestPATH);
          best_loss = Loss_val;
          early_stop_counter = 0;  
      }
      else
      {
          early_stop_counter++;
          if (early_stop_counter >= options.patience)
          {
              std::printf("\nEarly stopping at epoch %ld due to no improvement in validation loss.\n", epoch);
              break;
          }
      } 
    }
  
    logFile.close();

  }
  else
  {
    torch::jit::script::Module modelpred = torch::jit::load("/home/hmarefat/scratch/torchFOAM/JupyterLab/traced_model_M2_103.pt");
    modelpred.to(options.device);
    modelpred.to(torch::kDouble);
    printf("+-- The best model is loaded to the device!\n");
    
    std::ofstream fout("compare_training.dat");
    int count = 0;
    for(torch::data::Example<>& batch : *train_data_loader)
    {
      modelpred.eval();
      auto feat = batch.data.to(options.device);
      c10::IValue input = c10::IValue(feat); 
      auto labs = batch.target.to(options.device); 
      auto output = modelpred.forward({input});
      auto pred = output.toTensor();
    
      for(int i = 0; i < pred.sizes()[0]; i++)
      {
        fout << pred[i].item() << " \t" << labs[i].item() << "\n";
      }
      //std::cout << count << "/" << options.train_batches_per_epoch << "\n";
      //count++;
    }
    fout.close();
    
  }
  
  return 0;
}
