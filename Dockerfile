FROM ubuntu:22.04

# avoid time zone configuration by tzdata
ARG DEBIAN_FRONTEND=noninteractive

# install basic utilities
RUN apt-get update && apt-get install --no-install-recommends -y \
    software-properties-common \
    apt-utils       \
    add-apt-repository \
    ca-certificates \
    cmake           \
    g++             \
    make            \
    sudo            \
    git             \
    unzip           \
    vim-tiny        \
    wget            

# Install OpenFOAM v10 and modify it to use C++14: Just to make sure that the C++14 standard is used
ARG FOAM_PATH=/opt/openfoam10
RUN sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc" && \
    sudo add-apt-repository http://dl.openfoam.org/ubuntu && \
    sudo apt-get update && \
    sudo apt-get -y install openfoam10 && \
    echo ". ${FOAM_PATH}/etc/bashrc" >> /etc/bash.bashrc && \
    sed -i "s/-std=c++11/-std=c++14/g" ${FOAM_PATH}/wmake/rules/linux64Gcc/c++ && \
    sed -i "s/-Wold-style-cast/-Wno-old-style-cast/g" ${FOAM_PATH}/wmake/rules/linux64Gcc/c++


# download and extract the PyTorch C++ libraries (libtorch)
RUN wget -q -O libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip && \
    unzip libtorch.zip -d opt/ && \
    rm *.zip

## set libtorch enironment variable
ENV TORCH_LIBRARIES /opt/libtorch