#pragma once

#include <iostream>
#include <string>
#include <sstream>

struct Options 
{
    std::string file_path = " ";
    std::string file_name_without_extension = " ";
    int64_t MNum                = -3999;
    int64_t in_s                = 9;
    const int64_t hd_s          = 60;
    const int64_t hl_n          = 5;
    const int64_t ot_s          = 1;
    const int64_t patience      = 100;
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
    const bool shuffleDataset   = true;
    const int64_t numOfShuffle  = 10;
    const bool trainMode        = false;
    
    Options() 
    {
        std::cout << "Enter the file path: ";
        std::getline(std::cin, file_path);
        size_t found = file_path.find("/M");
        std::string tmp = file_path.substr(found + 1);
        size_t aa = tmp.find_last_of("_");
        tmp = tmp.substr(0, aa);
        std::stringstream conv2int;
        conv2int << tmp[1];
        conv2int >> MNum;
     
        size_t last_slash_position = file_path.find_last_of("/");
        std::string file_name = file_path.substr(last_slash_position + 1);
        size_t last_dot_position = file_name.find_last_of(".");
        file_name_without_extension = file_name.substr(0, last_dot_position);
    }
};
