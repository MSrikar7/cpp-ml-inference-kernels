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


## Continuous integration and reproducible benchmarking

[![C++ Kernel CI](https://github.com/MSrikar7/cpp-ml-inference-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/MSrikar7/cpp-ml-inference-kernels/actions/workflows/ci.yml)

Every push and pull request compiles the kernels with `g++ -O3 -std=c++17 -fopenmp`, verifies the numerical tolerance, and stores a benchmark CSV as a GitHub Actions artifact. CI verifies correctness but deliberately does not enforce a particular speedup on shared runners.

```bash
g++ -O3 -std=c++17 -fopenmp cpp/kernels.cpp -o kernels
python3 benchmarks/compare.py --binary ./kernels --repeats 7 --threads 4
```

The script saves `results/comparison.json` with identical matrix shapes and timed runs for naive, tiled, and parallel implementations.
