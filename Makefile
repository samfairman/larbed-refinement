CXX           = clang++
CXXFLAGS        = -std=c++14 -stdlib=libc++ -O3 -ferror-limit=2 -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-sign-conversion -Wno-exit-time-destructors -Wno-float-equal -Wno-global-constructors -Wno-missing-declarations -Wno-unused-parameter -Wno-padded -Wno-shadow -Wno-weak-vtables -Wno-missing-prototypes -Wno-unused-variable -ferror-limit=1 -Wno-deprecated -Wno-conversion -Wno-double-promotion
INCPATH       = -Iinclude -I/usr/local/cuda/include -I/usr/local/include -I/usr/include/c++/v1
LINK          = $(CXX)
LFLAGS        = -lc++ -lc++abi -O3 -pthread
DEL_FILE      = rm -f
DEL_DIR       = rmdir
MOVE          = mv -f
MAKE_DIR      = mkdir
CUDACXX      = nvcc
CUDACXX64      = nvcc
CUDACXXFLAGS = -m64 -gencode arch=compute_70,code=sm_70 -O2 -Iinclude
CUDALIB   = -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib 
CUDALFLAGS   = -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -lcudart -lcurand -lcublas
CUDALIB64   = -L/usr/local/cuda/lib64 
CUDALINK     = g++
LD_LIBRARY_PATH = /usr/local/cuda/lib64:/usr/local/lib64:/usr/lib64

####### Output directory
OBJECTS_DIR   = ./obj
BIN_DIR       = ./bin
LIB_DIR       = ./lib
LOG_DIR       = ./log

default: direct_cuda_pattern

direct_cuda_pattern: src/direct_cuda_pattern.cc
	$(CXX) $(CXXFLAGS) -c src/direct_cuda_pattern.cc -o $(OBJECTS_DIR)/direct_cuda_pattern.o $(INCPATH)
	$(CUDACXX64) src/cuda_pattern.cu -o $(OBJECTS_DIR)/_cuda_pattern.o $(INCPATH) -m64 -dc -gencode arch=compute_70,code=sm_70 --relocatable-device-code true  -Xptxas -v
	$(CUDACXX64) -dlink $(OBJECTS_DIR)/_cuda_pattern.o -arch=sm_70 -o $(OBJECTS_DIR)/cuda_pattern.o -m64 -lcudadevrt $(CUDALIB64) -rdc=true -Xptxas -v
	$(CXX) -o $(BIN_DIR)/direct_cuda_pattern $(OBJECTS_DIR)/_cuda_pattern.o $(OBJECTS_DIR)/direct_cuda_pattern.o $(OBJECTS_DIR)/cuda_pattern.o  -lcudadevrt -lcublas -lcudart -m64 $(CUDALIB64) $(LFLAGS)

