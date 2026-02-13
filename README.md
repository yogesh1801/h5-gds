# h5gds

* Simple benchmark code for GPUDirect Storage (GDS) via HDF5
* Developed by Yohei MIKI (Information Technology Center, The University of Tokyo)
* Released under the MIT license, see LICENSE for detail
* Copyright (c) 2023 Information Technology Center, The University of Tokyo

## How to compile

* Required software
  * [CMake](https://cmake.org/) (>= 3.20.0)
  * [Boost C++ Libraries](https://www.boost.org/)
  * [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (with GDS support)
  * [CUDA Samples](https://github.com/nvidia/cuda-samples)
    * set path of directly contains `helper_cuda.h` as `CPATH`, `CUDA_SAMPLES_DIR`, or `CUDA_SAMPLES_INC` for CMake
    * otherwise, edit `cmake/modules/FindCUDA_samples.cmake` properly
  * [HDF5](https://www.hdfgroup.org/solutions/hdf5/) (>= 1.14.0)
  * [HDF5 Nvidia GPUDirect Storage VFD](https://github.com/hpc-io/vfd-gds)
    * set path of directly contains `H5FDgds.h` as `CPATH`, `VFD_GDS_DIR`, or `VFD_GDS_INC` for CMake
    * set path of directly contains `libhdf5_vfd_gds.so` as `LD_LIBRARY_PATH`, `VFD_GDS_DIR`, or `VFD_GDS_LIB` for CMake
    * otherwise, edit `cmake/modules/FindHDF5VFD_GDS.cmake` properly
* Configuration using CUI

  ```sh
  cmake -S . -B build [option]
  cd build
  make
  ```

* Configuration using GUI

  ```sh
  cmake -S . -B build
  cd build
  ccmake -S .. # set options using the GUI interface
  make
  ```

* List of configure option(s)

  | input | note |
  | ---- | ---- |
  | `-DTARGET_GPU=[NVIDIA_CC90 NVIDIA_CC80 NVIDIA_CC86]` | target GPU architecture (e.g., NVIDIA_CC90 for NVIDIA H100) |

## How to run

* Execution

  ```sh
  bin/h5gds [option]
  ```

  * Execution in native mode (force to use GDS)

    ```sh
    CUFILE_JSON=./disable_compat.json bin/h5gds [option]
    ```

  * Execution in compatible mode (read/write via host CPU)

    ```sh
    CUFILE_JSON=./force_compat.json bin/h5gds [option]
    ```

* List of execution options
  * options have impact on benchmark score

    | input | note |
    | ---- | ---- |
    | `--asis` | adopt asis mode: read/write without hyperslab (i.e., disable hyperslab mode) |
    | `--num VALUE` | set VALUE as number of particles |
    | `--fblk VALUE` | set VALUE as file block size (byte) |
    | `--cbuf VALUE` | set VALUE as copy buffer size (byte); must be a multiple of the file block size |
    | `--memb VALUE` | set VALUE as memory boundary (byte) |

  * options have no impact on benchmark score

    | input | note |
    | ---- | ---- |
    | `--help` | show help message |
    | `--skip` | skip consistency check between read and original data |
    | `--virial VALUE` | set VALUE as the initial Virial ratio of the system |
    | `--radius VALUE` | set VALUE as the initial radius of the system |
    | `--mass VALUE` | set VALUE as the total mass of the system |
    | `--xdmf` | generate XDMF file to visualize the snapshot (only effective under hyperslab mode) |

## Data types

* $N$-element arrays to represent $N$-body particles

  | quantity | data on GPU | dataset in HDF5 file <br> (asis mode) | dataset in HDF5 file <br> (hyperslab mode) |
  | ---- | ---- | ---- | ---- |
  | position (x, y, z) <br> mass (w) | float4 pos[N] | float4 pos[N] | float position[N][3] <br> float mass[N] |
  | velocity (x, y) <br> velocity(z) | float2 vel_xy[N] <br> float vel_z[N] | float2 vel_xy[N] <br> float vel_z[N] | float velocity[N][3] |
  | particle ID | uint64_t idx[N] | uint64_t id[N] | uint64_t id[N] |



building in miyabi

```sh
cmake -S /work/jh250079/n14001/h5-gds \
      -B /work/jh250079/n14001/h5-gds/build \
      -DVFD_GDS_INC=/work/jh250079/n14001/vfd-gds/src \
      -DVFD_GDS_LIB=/work/jh250079/n14001/vfd-gds/build/bin \
      -DCUDA_SAMPLES_DIR=/work/jh250079/n14001/cuda-samples/Common \
      -DHDF5_ROOT=/work/jh250079/n14001/hdf5_install \
      -DCMAKE_PREFIX_PATH="/work/jh250079/n14001/netcdf_install;/work/jh250079/n14001/hdf5_install" \
      -DTARGET_GPU=NVIDIA_CC90 \
      -DFIRST_TOUCH_MODE=GPU



export CPATH=/work/jh250079/n14001/vfd-gds/src:$CPATH
export CPATH=/work/jh250079/n14001/netcdf_install/include:$CPATH
export LIBRARY_PATH=/work/jh250079/n14001/netcdf_install/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/work/jh250079/n14001/vfd-gds/build/bin:$LD_LIBRARY_PATH


```


```sh
https://chatgpt.com/share/68e69999-0e84-800e-a59f-d8042f5a9053 this is for colva
```


```sh

for tsubame

cmake -S . -B build \
  -DVFD_GDS_INC=/work/1/uu05721/vfd-gds/src \
  -DVFD_GDS_LIB=/work/1/uu05721/vfd-gds/build/bin \
  -DCUDA_SAMPLES_DIR=/work/1/uu05721/cuda-samples/Common \
  -DBOOST_ROOT=/work/1/uu05721/boost-install \
  -DBOOST_INCLUDEDIR=/work/1/uu05721/boost-install/include \
  -DBOOST_LIBRARYDIR=/work/1/uu05721/boost-install/lib \
  -DBoost_NO_SYSTEM_PATHS=ON \
  -DBoost_DEBUG=ON

```
