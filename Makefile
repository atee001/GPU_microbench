# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++11 -Wall -Wextra -Iinclude

# Directories
SRC_DIR := src
OBJ_DIR := build
BIN_DIR := bin
ASM_DIR := asm   # new directory for assembly files

# Source and object files
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))
ASMS := $(patsubst $(SRC_DIR)/%.cpp,$(ASM_DIR)/%.s,$(SRCS))  # new assembly files

# Target executable
TARGET := $(BIN_DIR)/benchmark

# Default target
all: $(TARGET)

# Link the final executable
$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Compile source files into object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Generate assembly files
$(ASM_DIR)/%.s: $(SRC_DIR)/%.cpp
	@mkdir -p $(ASM_DIR)
	$(CXX) $(CXXFLAGS) -S $< -o $@

# Optional: generate all assembly files
asm: $(ASMS)

# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(ASM_DIR)

.PHONY: all clean asm
