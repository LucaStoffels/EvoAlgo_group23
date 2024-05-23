CC=gcc
RM=rm
BUILD_DIR := build
SRC_DIR := src
INC_DIR = include
FLAGS=-Wall -I$(INC_DIR)
CFLAGS=$(FLAGS) -O3 -lm
CFLAGS_DEBUG=$(FLAGS) -pg -lm
LDFLAGS = -g
LDFLAGS_DEBUG = -pg
LIBS =

SHARED_SOURCE = $(wildcard $(SRC_DIR)/shared/*.cpp)
SHARED_OBJECTS = $(SHARED_SOURCE:.cpp=.o)
SHARED_OBJECTS = $(SHARED_SOURCE:%.cpp=$(BUILD_DIR)/%.o)
SHARED_DEBUG_OBJECTS = $(SHARED_OBJECTS:$(BUILD_DIR)/src/%.cpp=$(BUILD_DIR)/debug/%.o)

.PHONY: all
all: clean builddir	full univariate

clean:
	$(RM) -f $(SHARED_OBJECTS) $(SHARED_DEBUG_OBJECTS) build/src/*.o build/debug/*.o

builddir:
	mkdir -p build/src
	mkdir -p build/debug
	mkdir -p build/src/shared
	mkdir -p build/debug/shared

full-debug: build/debug/amalgam_full.o $(SHARED_DEBUG_OBJECTS)
	$(CC) $(LDFLAGS_DEBUG) $(^:debug/=src/) -o $@ $(LIBS)

full: build/src/amalgam_full.o $(SHARED_OBJECTS)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS) -lm

univariate-debug: build/debug/amalgam_univariate.o $(SHARED_DEBUG_OBJECTS)
	$(CC) $(LDFLAGS_DEBUG) $(^:debug/=src/) -o $@ $(LIBS)

univariate: build/src/amalgam_univariate.o $(SHARED_OBJECTS)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS) -lm

$(BUILD_DIR)/debug/shared/%.o: src/shared/%.cpp
	$(CC) -c $(CFLAGS_DEBUG) -o $@ $<

$(BUILD_DIR)/src/shared/%.o: src/shared/%.cpp
	$(CC) -c $(CFLAGS) -o $@ $<

$(BUILD_DIR)/debug/%.o: src/%.cpp
	$(CC) -c $(CFLAGS_DEBUG) -o $@ $<

$(BUILD_DIR)/src/%.o: src/%.cpp
	@echo $(CFLAGS)
	$(CC) -c $(CFLAGS) -o $@ $<