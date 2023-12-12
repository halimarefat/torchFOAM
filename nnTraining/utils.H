#pragma once

#include <iostream>
#include <iomanip>
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

  double t, u, v, w, G1, G2, G3, G4, G5, G6, s1, s2, s3, s4, s5, s6, up1, up2, up3, up4, up5, up6, cs;
  
  std::string line;
  std::getline (fistream, line);
  
  while(fistream >> t)
  {
    std::vector<double> tmp_inp;
    fistream >> u >> v >>	w >>	
          G1 >>	G2 >> G3 >>	G4 >> G5 >>	G6 >> 
          s1 >>	s2 >> s3 >>	s4 >> s5 >>	s6 >> 
          up1 >> up2 >> up3 >> up4 >> up5 >> up6 >>	
          cs;

    if (options.MNum == 1)
    {
        options.in_s = 9;
        tmp_inp.push_back(u);
        tmp_inp.push_back(v);
        tmp_inp.push_back(w);
        tmp_inp.push_back(s1);
        tmp_inp.push_back(s2);
        tmp_inp.push_back(s3);
        tmp_inp.push_back(s4);
        tmp_inp.push_back(s5);
        tmp_inp.push_back(s6);
    }
    else if (options.MNum == 2)
    {
        options.in_s = 12;
        tmp_inp.push_back(G1);
        tmp_inp.push_back(G2);
        tmp_inp.push_back(G3);
        tmp_inp.push_back(G4);
        tmp_inp.push_back(G5);
        tmp_inp.push_back(G6);
        tmp_inp.push_back(s1);
        tmp_inp.push_back(s2);
        tmp_inp.push_back(s3);
        tmp_inp.push_back(s4);
        tmp_inp.push_back(s5);
        tmp_inp.push_back(s6);
    }
    else if (options.MNum == 3)
    {
        options.in_s = 9;
        tmp_inp.push_back(u);
        tmp_inp.push_back(v);
        tmp_inp.push_back(w);
        tmp_inp.push_back(up1);
        tmp_inp.push_back(up2);
        tmp_inp.push_back(up3);
        tmp_inp.push_back(up4);
        tmp_inp.push_back(up5);
        tmp_inp.push_back(up6);
    }
    else if (options.MNum == 4)
    {
        options.in_s = 12;
        tmp_inp.push_back(G1);
        tmp_inp.push_back(G2);
        tmp_inp.push_back(G3);
        tmp_inp.push_back(G4);
        tmp_inp.push_back(G5);
        tmp_inp.push_back(G6);
        tmp_inp.push_back(up1);
        tmp_inp.push_back(up2);
        tmp_inp.push_back(up3);
        tmp_inp.push_back(up4);
        tmp_inp.push_back(up5);
        tmp_inp.push_back(up6);
    }
    else if (options.MNum == 5)
    {
        options.in_s = 22;
        tmp_inp.push_back(t);
        tmp_inp.push_back(u);
        tmp_inp.push_back(v);
        tmp_inp.push_back(w);
        tmp_inp.push_back(G1);
        tmp_inp.push_back(G2);
        tmp_inp.push_back(G3);
        tmp_inp.push_back(G4);
        tmp_inp.push_back(G5);
        tmp_inp.push_back(G6);
        tmp_inp.push_back(up1);
        tmp_inp.push_back(up2);
        tmp_inp.push_back(up3);
        tmp_inp.push_back(up4);
        tmp_inp.push_back(up5);
        tmp_inp.push_back(up6);
        tmp_inp.push_back(s1);
        tmp_inp.push_back(s2);
        tmp_inp.push_back(s3);
        tmp_inp.push_back(s4);
        tmp_inp.push_back(s5);
        tmp_inp.push_back(s6);
    }
    else
    {

    }
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

  std::ofstream train_fout("train.dat");
  std::ofstream val_fout("val.dat");
  for(auto r : train)
  {
    if (options.in_s == 9)
    {
        train_fout << std::setprecision(15) << r.first[0]  << " " << r.first[1]  << " " << r.first[2] << " " << r.first[3] << " " << r.first[4] 
            << " " << r.first[5]  << " " << r.first[6]  << " " << r.first[7] << " " << r.first[8] << " "<< r.second
            << "\n";
    }
    else if (options.in_s == 12)
    {
        train_fout << std::setprecision(15) << r.first[0]  << " " << r.first[1]  << " " << r.first[2] << " " << r.first[3] << " " << r.first[4] 
            << " " << r.first[5]  << " " << r.first[6]  << " " << r.first[7] << " " << r.first[8] << " " << r.first[9] 
            << " " << r.first[10] << " " << r.first[11] << " "<< r.second
            << "\n";
    }
    else if (options.in_s == 22)
    {
        train_fout << std::setprecision(15) << r.first[0]  << " " << r.first[1]  << " " << r.first[2] << " " << r.first[3] << " " << r.first[4] 
            << " " << r.first[5]  << " " << r.first[6]  << " " << r.first[7] << " " << r.first[8] << " " << r.first[9] 
            << " " << r.first[10] << " " << r.first[11] << " "<< r.first[12] << " "<< r.first[13] << " "<< r.first[14] << " "<< r.first[15] 
            << " " << r.first[16] << " "<<r.first[17] << " "<< r.first[18] << " "<< r.first[19] << " "<< r.first[20] << " "<< r.first[21] << " "<< r.second
            << "\n";
    }
  }
  for(auto r : val)
  {
    if (options.in_s == 9)
    {
        val_fout << std::setprecision(15) << r.first[0]  << " " << r.first[1]  << " " << r.first[2] << " " << r.first[3] << " " << r.first[4] 
            << " " << r.first[5]  << " " << r.first[6]  << " " << r.first[7] << " " << r.first[8] << " "<< r.second
            << "\n";
    }
    else if (options.in_s == 12)
    {
        val_fout << std::setprecision(15) << r.first[0]  << " " << r.first[1]  << " " << r.first[2] << " " << r.first[3] << " " << r.first[4] 
            << " " << r.first[5]  << " " << r.first[6]  << " " << r.first[7] << " " << r.first[8] << " " << r.first[9] 
            << " " << r.first[10] << " " << r.first[11] << " "<< r.second
            << "\n";
    }
    else if (options.in_s == 22)
    {
        val_fout << std::setprecision(15) << r.first[0]  << " " << r.first[1]  << " " << r.first[2] << " " << r.first[3] << " " << r.first[4] 
            << " " << r.first[5]  << " " << r.first[6]  << " " << r.first[7] << " " << r.first[8] << " " << r.first[9] 
            << " " << r.first[10] << " " << r.first[11] << " "<< r.first[12] << " "<< r.first[13] << " "<< r.first[14] << " "<< r.first[15] 
            << " " << r.first[16] << " "<<r.first[17] << " "<< r.first[18] << " "<< r.first[19] << " "<< r.first[20] << " "<< r.first[21] << " "<< r.second
            << "\n";
    }
  }
  
  train_fout.close();
  val_fout.close();

  return std::make_pair(train, val);
}