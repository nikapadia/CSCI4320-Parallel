A3: A3.o
  mpixlc A3.o -o A3 \
  -L/usr/local/cuda-11.2/lib64 \
  -lcudart -lcudadevrt -lcurand -lstdc++ \
  -L/opt/ibm/spectrum_mpi/lib -lmpi_ibm

A3.o: A3.cu
  nvcc -O3 -arch=sm_70 -I/opt/ibm/spectrum_mpi/include -c A3.cu -o A3.o