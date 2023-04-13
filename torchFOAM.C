#include <torch/torch.h>
#include <iostream>
#include <memory>

#include "IncompressibleTurbulenceModel.H"
#include "incompressible/transportModel/transportModel.H"
#include "addToRunTimeSelectionTable.H"
#include "makeTurbulenceModel.H"

#include "laminarModel.H"
#include "RASModel.H"
#include "LESModel.H"

struct DCGANGeneratorImpl : torch::nn::Module 
{
  DCGANGeneratorImpl(int kNoiseSize)
      : conv1(torch::nn::ConvTranspose2dOptions(kNoiseSize, 256, 4).bias(false)),
        batch_norm1(256),
        conv2(torch::nn::ConvTranspose2dOptions(256, 128, 3).stride(2).padding(1).bias(false)),
        batch_norm2(128),
        conv3(torch::nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1).bias(false)),
        batch_norm3(64),
        conv4(torch::nn::ConvTranspose2dOptions(64, 1, 4).stride(2).padding(1).bias(false))
  {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("conv4", conv4);
    register_module("batch_norm1", batch_norm1);
    register_module("batch_norm2", batch_norm2);
    register_module("batch_norm3", batch_norm3);
  }

  torch::Tensor forward(torch::Tensor x) 
  {
    x = torch::relu(batch_norm1(conv1(x)));
    x = torch::relu(batch_norm2(conv2(x)));
    x = torch::relu(batch_norm3(conv3(x)));
    x = torch::tanh(conv4(x));
    return x;
  }

  torch::nn::ConvTranspose2d conv1, conv2, conv3, conv4;
  torch::nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
};
TORCH_MODULE(DCGANGenerator);


struct DCGANDiscriminatorImpl : torch::nn::Module 
{
  DCGANDiscriminatorImpl(int in)
      : conv1(torch::nn::Conv2dOptions(in, 64, 4).stride(2).padding(1).bias(false)),
        batch_norm1(64),
        Lrelu1(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
        conv2(torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
        batch_norm2(128),
        Lrelu2(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
        conv3(torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
        batch_norm3(256),
        Lrelu3(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
        conv4(torch::nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false))
  {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("conv4", conv4);
    register_module("batch_norm1", batch_norm1);
    register_module("batch_norm2", batch_norm2);
    register_module("batch_norm3", batch_norm3);
  }

  torch::Tensor forward(torch::Tensor x) 
  {
    x = Lrelu1(batch_norm1(conv1(x)));
    x = Lrelu2(batch_norm2(conv2(x)));
    x = Lrelu3(batch_norm3(conv3(x)));
    x = torch::sigmoid(conv4(x));
    return x;
  }

  torch::nn::Conv2d conv1, conv2, conv3, conv4;
  torch::nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
  torch::nn::LeakyReLU Lrelu1, Lrelu2, Lrelu3;
};
TORCH_MODULE(DCGANDiscriminator);


int main() 
{
  std::cout << "PyTorch version: "
  << TORCH_VERSION_MAJOR << "."
  << TORCH_VERSION_MINOR << "."
  << TORCH_VERSION_PATCH << std::endl;

  torch::Device device(torch::kCPU);
  std::cout << "CUDA DEVICE COUNT: " << torch::cuda::device_count() << std::endl;
  if (torch::cuda::is_available()) 
  {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::kCUDA;
  }

  int kNoiseSize = 100;
  int64_t kBatchSize = 64;
  int64_t kNumberOfEpochs = 50;
  int in = 1;

  DCGANGenerator generator(kNoiseSize);
  DCGANDiscriminator discriminator(in);

  auto dataset = torch::data::datasets::MNIST("./mnist")
    .map(torch::data::transforms::Normalize<>(0.5, 0.5))
    .map(torch::data::transforms::Stack<>());
  
  auto data_loader = torch::data::make_data_loader(
    std::move(dataset),
    torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));

  
  int count = 0;
  for (torch::data::Example<>& batch : *data_loader) 
  {
    std::cout << count++ << " " << "Batch size: " << batch.data.size(0) << " | Labels: ";
    for (int64_t i = 0; i < batch.data.size(0); ++i) 
    {
      std::cout << batch.target[i].item<int64_t>() << " ";
    }
    std::cout << std::endl;
  }
  int64_t batches_per_epoch = count;

  torch::optim::Adam generator_optimizer(generator->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.5)));
  torch::optim::Adam discriminator_optimizer(generator->parameters(), torch::optim::AdamOptions(5e-4).betas(std::make_tuple(0.5, 0.5)));

  generator->to(device);
  discriminator->to(device);

  for(int64_t epoch=1; epoch <= kNumberOfEpochs; ++epoch)
  {
    int64_t batch_index = 0;

    for(torch::data::Example<>& batch : *data_loader)
    {
      discriminator->zero_grad();
      torch::Tensor real_images = batch.data.to(device);
      torch::Tensor real_labels = torch::empty(batch.data.size(0)).uniform_(0.8, 1.0).to(device);
      torch::Tensor real_output = discriminator->forward(real_images);
      torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
      d_loss_real.backward();

      torch::Tensor noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1}).to(device);
      torch::Tensor fake_images = generator->forward(noise);
      torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);
      torch::Tensor fake_output = discriminator->forward(fake_images.detach());
      torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
      d_loss_fake.backward();

      torch::Tensor d_loss = d_loss_fake + d_loss_real;
      discriminator_optimizer.step();

      generator->zero_grad();
      fake_labels.fill_(1);
      fake_output = discriminator->forward(fake_images.detach());
      torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
      g_loss.backward();
      generator_optimizer.step();

      std::printf("\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f",
        epoch,
        kNumberOfEpochs,
        ++batch_index,
        batches_per_epoch,
        d_loss.item<float>(),
        g_loss.item<float>());
    }
  }

  return 0;
}
