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

  double G1, G2, G3, G4, G5, G6, up1,up2,up3,up4,up5,up6, cs;
  //double u,v,w,up1,up2,up3,up4,up5,up6,cs;
  //double u,v,w,s1,s2,s3,s4,s5,s6,cs;
  //double up1,up2,up3,up4,up5,up6,s1,s2,s3,s4,s5,s6,cs;
  while(fistream >> G1)
  {
    std::vector<double> tmp_inp;
    fistream >> G2  >> G3  >> G4  >> G5  >> G6  
             >> up1 >> up2 >> up3 >> up4 >> up5 >> up6
             >> cs;
    //fistream >> v   >> w   >> up1 >> up2 
    //         >> up3 >> up4 >> up5 >> up6
    //         >> cs; 
    //fistream >> v >> w >> s1 >> s2 
    //         >> s3 >> s4 >> s5 >> s6
    //         >> cs; 
    //fistream >> up1 >> up2 >> up3 >> up4 >> up5 >> up6 >>
    //            s1  >> s2  >> s3  >> s4  >> s5  >> s6  >> 
    //            cs; 
    //tmp_inp.push_back(u);
    //tmp_inp.push_back(v);
    //tmp_inp.push_back(w);
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
    //tmp_inp.push_back(s1);
    //tmp_inp.push_back(s2);
    //tmp_inp.push_back(s3);
    //tmp_inp.push_back(s4);
    //tmp_inp.push_back(s5);
    //tmp_inp.push_back(s6);
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
    train_fout << std::setprecision(15) << r.first[0]  << " " << r.first[1]  << " " << r.first[2] << " " << r.first[3] << " " << r.first[4] 
        << " " << r.first[5]  << " " << r.first[6]  << " " << r.first[7] << " " << r.first[8] << " " << r.first[9] 
        << " " << r.first[10] << " " << r.first[11] << " "<< r.second
        
        //<< r.first[9]
        //<< " " << r.first[10] << " " << r.first[11] << " " << r.second 
        << "\n";

  }
  for(auto r : val)
  {
    val_fout   << std::setprecision(15) << r.first[0]  << " " << r.first[1]  << " " << r.first[2] << " " << r.first[3] << " " << r.first[4] 
        << " " << r.first[5]  << " " << r.first[6]  << " " << r.first[7] << " " << r.first[8] << " " << r.first[9] 
        << " " << r.first[10] << " " << r.first[11] << " "<< r.second
        //<< r.first[9]
        //<< " " << r.first[10] << " " << r.first[11] << " " << r.second 
        << "\n";

  }
  
  train_fout.close();
  val_fout.close();

  return std::make_pair(train, val);
}