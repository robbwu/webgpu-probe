# Compiler settings
CXX = clang++
OBJCXX = clang++

# File names
SOURCE = bench.mm
TARGET = bench

# Frameworks and include paths for macOS/Metal
FRAMEWORKS = -framework Metal -framework MetalPerformanceShaders -framework Foundation
INCLUDES = -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Metal.framework/Headers \
          -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/MetalPerformanceShaders.framework/Headers

# Compiler flags
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
OBJCXXFLAGS = $(CXXFLAGS) -fobjc-arc

# Linker flags
LDFLAGS = $(FRAMEWORKS)

# Build rules
all: $(TARGET)

$(TARGET): $(SOURCE)
	$(OBJCXX) $(OBJCXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

# Clean rule
clean:
	rm -f $(TARGET)

# Debug build with symbols
debug: CXXFLAGS += -g -DDEBUG
debug: OBJCXXFLAGS += -g -DDEBUG
debug: all

.PHONY: all clean debug
