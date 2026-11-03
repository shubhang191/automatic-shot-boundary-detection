CXX ?= g++
FLAGS = -O3 -std=c++17
all: bin/sbdv bin/libsbdv.so
bin/sbdv: src/sbdv.hpp src/sbdv_cli.cpp
	mkdir -p bin && $(CXX) $(FLAGS) -Iinclude -Isrc src/sbdv_cli.cpp -o $@
bin/libsbdv.so: src/sbdv.hpp src/sbdv_capi.cpp include/sbdv.h
	mkdir -p bin && $(CXX) $(FLAGS) -shared -fPIC -fvisibility=hidden -Iinclude -Isrc src/sbdv_capi.cpp -o $@
clean:
	rm -rf bin
