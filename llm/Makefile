
TARGET = exponentialIntegral
SRC    = main.cu
ARCH   = -gencode arch=compute_75,code=sm_75

all:
	nvcc $(SRC) -o $(TARGET) $(ARCH) -O3

clean:
	rm -f $(TARGET)
