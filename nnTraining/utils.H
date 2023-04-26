#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include "config.H"

using DatasetType = std::vector<std::pair<std::vector<double>, double>>;

std::pair<DatasetType, DatasetType> datasetCreator(Options& options) 
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