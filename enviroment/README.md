# Environment Docker Image

This folder contains the `Dockerfile` that sets up all dependencies (Clang, LLVM, Enzyme, Eigen, OpenMP, CUDA dev libraries).

## Build the image
From the repository root:
```bash
docker build -t enzyme-image enviroment
```

## Run the container
Mount the repository and work inside the container:
```bash
docker run --rm -it -v "$(pwd)":/work -w /work enzyme-image /bin/bash
```

## Build and run inside the container
CPU build:
```bash
mkdir -p build
cd build
cmake .. -DENABLE_CUDA=OFF
make
```

CPU + CUDA build:
```bash
mkdir -p build
cd build
cmake .. -DENABLE_CUDA=ON
make
```

## Burgers test (Enzyme)
```bash
cmake -S tests/burgers -B build-burgers
cmake --build build-burgers -j
./build-burgers/test_burgers_parallel
```
