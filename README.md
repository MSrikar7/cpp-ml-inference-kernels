# High-Performance C++ ML Compute Kernels

C++17 and OpenMP CPU matrix multiplication and bias kernel benchmarks exploring cache tiling and thread parallelism. This is an independent kernel benchmark, not a custom PyTorch operator or end-to-end ML inference deployment.

## Run

```bash
g++ -O3 -std=c++17 -fopenmp cpp/kernels.cpp -o kernels
./kernels 128 256 128 32 7 4
```

## Results

See `results/cpp_benchmark.csv` for recorded local results. Re-run on your own machine before making performance claims. Timings depend on CPU, compiler, and thread configuration.

## Limitations

No GPU/CUDA, ONNX integration, MLIR, or production-scale inference is claimed.
