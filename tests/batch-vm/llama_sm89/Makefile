NVCC?=nvcc
TARGET=kvm_llama_sm89
SRC=llama_sm89.cu

ifndef PYTHON_VERSION
PYTHON_VERSION=3.12
endif

THUNDERKITTENS_ROOT=$(shell pwd)/../../../

NVCCFLAGS = -DNDEBUG \
            -DKITTENS_4090 \
            -Xcompiler=-fPIE \
            --expt-extended-lambda \
            --expt-relaxed-constexpr \
            -Xcompiler=-Wno-psabi \
            -Xcompiler=-fno-strict-aliasing \
            --use_fast_math \
            -forward-unknown-to-host-compiler \
            -O3 \
            -Xnvlink=--verbose \
            -Xptxas=--verbose \
            -Xptxas=--warn-on-spills \
            -std=c++20 \
            -x cu \
            -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static -lcublas \
            -lineinfo \
            -arch=sm_89 \
            -I${THUNDERKITTENS_ROOT}/include \
            -I${THUNDERKITTENS_ROOT}/prototype \
            $(shell python3 -m pybind11 --includes) \
            $(shell python3-config --ldflags) \
            -shared -fPIC \
            -lpython${PYTHON_VERSION}

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)$(shell python3-config --extension-suffix)

clean:
	rm -f $(TARGET)$(shell python3-config --extension-suffix)
