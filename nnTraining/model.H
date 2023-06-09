#pragma once

#include <torch/torch.h>

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

class newNNImpl : public torch::nn::Module 
{
  torch::nn::Linear lin1, lin2, lin3, lin4, lin5, lin6;
  
public:
  newNNImpl(const int64_t in_size, const int64_t hid_size, const int64_t out_size)
    : lin1(torch::nn::LinearOptions(in_size , hid_size)),
      lin2(torch::nn::LinearOptions(hid_size, hid_size)),
      lin3(torch::nn::LinearOptions(hid_size, hid_size)),
      lin4(torch::nn::LinearOptions(hid_size, hid_size)),
      lin5(torch::nn::LinearOptions(hid_size, hid_size)),
      lin6(torch::nn::LinearOptions(hid_size, out_size)) 
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
    x = torch::relu(lin2->forward(x));
    x = torch::relu(lin3->forward(x)); 
    x = torch::relu(lin4->forward(x)); 
    x = torch::relu(lin5->forward(x));  
    x = lin6->forward(x);
    return x;
  }
};
TORCH_MODULE(newNN);

class NN2Impl : public torch::nn::Module 
{
  torch::nn::Linear lin1, lin2, lin3, lin4, lin5, lin6;
  
public:
  NN2Impl(const int64_t in_size, const int64_t hid_size, const int64_t out_size)
    : lin1(torch::nn::LinearOptions(in_size , hid_size)),
      lin2(torch::nn::LinearOptions(hid_size, hid_size)),
      lin3(torch::nn::LinearOptions(hid_size, hid_size)),
      lin4(torch::nn::LinearOptions(hid_size, hid_size-3)),
      lin5(torch::nn::LinearOptions(hid_size-3, hid_size-4)),
      lin6(torch::nn::LinearOptions(hid_size-4, out_size)) 
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
    x = torch::relu(lin2->forward(x));
    x = torch::relu(lin3->forward(x));
    //x = torch::dropout(x, 0.1, is_training()); 
    x = torch::relu(lin4->forward(x));
    //x = torch::dropout(x, 0.1, is_training()); 
    x = torch::relu(lin5->forward(x));   
    x = lin6->forward(x);
    return x;
  }
};
TORCH_MODULE(NN2);

class NN1Impl : public torch::nn::Module 
{
  torch::nn::Linear lin1, lin2, lin3, lin4, lin5, lin6;
  torch::nn::BatchNorm1d btn1, btn2, btn3, btn4, btn5;
  
public:
  NN1Impl(const int64_t in_size, const int64_t hid_size, const int64_t out_size)
    : lin1(torch::nn::LinearOptions(in_size , hid_size)),
      btn1(torch::nn::BatchNorm1d(hid_size)),
      lin2(torch::nn::LinearOptions(hid_size, hid_size)),
      btn2(torch::nn::BatchNorm1d(hid_size)),
      lin3(torch::nn::LinearOptions(hid_size, hid_size)),
      btn3(torch::nn::BatchNorm1d(hid_size)),
      lin4(torch::nn::LinearOptions(hid_size, hid_size)),
      btn4(torch::nn::BatchNorm1d(hid_size)),
      lin5(torch::nn::LinearOptions(hid_size, hid_size)),
      btn5(torch::nn::BatchNorm1d(hid_size)),
      lin6(torch::nn::LinearOptions(hid_size, out_size)) 
  {
    register_module("linear1", lin1);
    register_module("linear2", lin2);
    register_module("linear3", lin3);
    register_module("linear4", lin4);
    register_module("linear5", lin5);
    register_module("linear6", lin6);
    register_module("batch_norm1", btn1);
    register_module("batch_norm2", btn2);
    register_module("batch_norm3", btn3);
    register_module("batch_norm4", btn4);
    register_module("batch_norm5", btn5);
  }

  torch::Tensor forward(torch::Tensor& x) 
  {
    x = torch::relu(btn1(lin1->forward(x)));
    x = torch::dropout(x, 0.1, is_training());
    x = torch::relu(btn2(lin2->forward(x)));
    x = torch::dropout(x, 0.1, is_training());
    x = torch::relu(btn3(lin3->forward(x))); 
    x = torch::dropout(x, 0.1, is_training());
    x = torch::relu(btn4(lin4->forward(x))); 
    x = torch::dropout(x, 0.1, is_training());
    x = torch::relu(btn5(lin5->forward(x)));  
    x = torch::dropout(x, 0.1, is_training());
    x = lin6->forward(x);
    return x;
  }
};
TORCH_MODULE(NN1);