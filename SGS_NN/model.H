#pragma once

#include <torch/torch.h>

class ANNModelImpl : public torch::nn::Module 
{
public:
    ANNModelImpl(int64_t in_size, int64_t out_size, int64_t hl_num, int64_t hid_size)
        : in(register_module("Input", torch::nn::Linear(in_size, hid_size))),
          out(register_module("Output", torch::nn::Linear(hid_size, out_size))) 
    {
        for (int64_t i = 0; i < hl_num; ++i) 
        {
            auto l = register_module("hidden" + std::to_string(i), torch::nn::Linear(hid_size, hid_size));
            hidden_layers->push_back(l);
        }
    }

    torch::Tensor forward(torch::Tensor x) 
    {
        x = torch::relu(in->forward(x));
        for (int64_t i = 0; i < hidden_layers->size(); ++i) {
            x = torch::relu((*hidden_layers)[i]->as<torch::nn::LinearImpl>()->forward(x));
        }
        x = out->forward(x);
        return x;
    }

private:
    torch::nn::Linear in, out;
    torch::nn::ModuleList hidden_layers;
};
TORCH_MODULE(ANNModel);


class ANNModel_DP : public torch::nn::Module 
{ 
public:
  ANNModel_DP(const int64_t in_size, const int64_t hid_size, const int64_t out_size)
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
private:
  torch::nn::Linear lin1, lin2, lin3, lin4, lin5, lin6;
};


class ANNModel_DP_BTN : public torch::nn::Module 
{
public:
  ANNModel_DP_BTN(const int64_t in_size, const int64_t hid_size, const int64_t out_size)
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

private:
  torch::nn::Linear lin1, lin2, lin3, lin4, lin5, lin6;
  torch::nn::BatchNorm1d btn1, btn2, btn3, btn4, btn5;
};
