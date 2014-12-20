##################################
# Set path to dependencies.
INCLUDE := -I../deps/eigen
###################################

CCFLAGS := -O3 -std=c++0x -fPIC -shared
LIB_NAME := libeigenmat.so

all: eigenmat.cc eigenmat.h
	g++ $(CCFLAGS) $(INCLUDE) -o $(LIB_NAME) eigenmat.cc

clean:
	rm -rf $(LIB_NAME)
