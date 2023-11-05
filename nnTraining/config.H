#pragma once

#include <iostream>

struct Options 
{
  const std::string file_path = "/home/hmarefat/scratch/torchFOAM/datasetGen/dSHR_BBPF_GUp_150_160__norm.npy";
  const int64_t in_s          = 12;
  const int64_t hd_s          = 60;
  const int64_t hl_n          = 5;
  const int64_t ot_s          = 1;
  const int64_t numbOfEpochs  = 3000;
  const int64_t batchSize     = 4096; //2048; //1028; //524288; // size of one time-step >> 524288;
  const int64_t valBatchSize  = 1024; //512; //257;
  int64_t dataset_size        = 0;
  const float train_val_split = 0.2;
  const float test_split      = 0.1;
  int64_t train_size          = 0;
  int64_t val_size            = 0;
  int64_t train_batches_per_epoch = 0;
  int64_t val_batches_per_epoch   = 0;
  torch::DeviceType device    = torch::kCPU;
  const bool printBatches     = false;
  const bool shuffleDataset   = false;
  const int64_t numOfShuffle  = 10;
  const bool trainMode        = true;
};
