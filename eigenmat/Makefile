##################################
# Set path to dependencies.
INCLUDE := -I../deps/eigen
###################################

CCFLAGS := -O3 -fPIC -shared
LIB_NAME := libeigenmat.so

all: eigenmat.cc eigenmat.h ziggurat.cc ziggurat.h
	g++ $(CCFLAGS) $(INCLUDE) -o $(LIB_NAME) eigenmat.cc ziggurat.cc

clean:
	rm -rf $(LIB_NAME)

