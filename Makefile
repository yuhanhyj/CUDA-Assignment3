CC=g++
CFLAGS= -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops

NVCC=nvcc
NVCCFLAGS=-O3 -arch=compute_75 -code=sm_75 -lcudart

INCPATH       = -I.

TARGET=main.o exponentialIntegralGPU.o
EXEC=exponentialIntegral.out


all: $(TARGET)
	$(NVCC) ${NVCCFLAGS} -o ${EXEC} ${TARGET}

%.o: %.cpp
	$(CC) $(CFLAGS) -c $(INCPATH) $<

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $(INCPATH) $<

install:

clean:
	rm -f *.o ${TARGET} ${EXEC}
