# High Performance Seismic Stacking

This projects aims to provide a high performance implementation for Seismic Stack computing method with support to three diferent traveltime models:

* **Common Mid Point**
* **Zero Offset Common Reflection Surface**
* **Offset Continuation Trajectory**

It's provided an implementation for both CUDA and OpenCL (in development) frameworks.

An implementation to get rid of stretch behavior is also available.

## Dependencies

* `gcc` >= 8.4.0
* `nvcc` >= 11.0.194 for `CUDA`
* `CUDA` `drivers` >= 11
* `OpenCL` >= 1.2
* `libboost` >= 1.65.01
* `c++17`

## Running

To further information on how to run the implementations of this project, refer to `cuda/` or `opencl/` according to your needs.

## License

This project is provided under GPLv3 license.
