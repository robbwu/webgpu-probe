CXX = clang++
CXXFLAGS = -std=c++11 -O3 -Wall
INCLUDES =   -I.
LDFLAGS = -L .
LIBS = -framework OpenCL -lclblast

TARGET = bench
SRCS = bench.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS) $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
