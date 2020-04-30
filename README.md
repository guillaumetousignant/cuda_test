# cuda_test

Installation:

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_CUDA_FLAGS=”-arch=sm_61”
make
make install
```