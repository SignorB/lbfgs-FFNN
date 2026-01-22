FROM docker.io/nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    cmake \
    ninja-build \
    python3 \
    python3-pip \
    zlib1g-dev \
    libeigen3-dev \
    clang-15 \
    llvm-15 \
    llvm-15-dev \
    libclang-15-dev \
    libomp-15-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir lit

WORKDIR /usr/src
RUN git clone https://github.com/EnzymeAD/Enzyme.git


WORKDIR /usr/src/Enzyme/enzyme
RUN mkdir build
WORKDIR /usr/src/Enzyme/enzyme/build

RUN cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_DIR=/usr/lib/llvm-15/lib/cmake/llvm \
    -DLLVM_EXTERNAL_LIT=/usr/local/bin/lit

RUN ninja

# RUN ninja check-enzyme

ENV CC=clang-15
ENV CXX=clang++-15

CMD ["/bin/bash"]
