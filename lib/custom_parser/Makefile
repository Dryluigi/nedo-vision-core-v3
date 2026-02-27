CC:= g++

CFLAGS:= -O3 -Wall -std=c++11 -shared -fPIC

LIBS:= -lnvinfer -lnvparsers -L/opt/nvidia/deepstream/deepstream/lib

INCLUDES:= -I /opt/nvidia/deepstream/deepstream/sources/includes \
           -I /usr/local/cuda/include

TARGET_LIB:= libcustom_parser.so

all: $(TARGET_LIB)

$(TARGET_LIB): custom_parser.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

clean:
	rm -rf $(TARGET_LIB)
