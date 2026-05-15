BUILDDIR=build
CC=clang++
CFLAGS=-Wall -Werror -Wextra -std=c++23 -I.
TEST_CFLAGS=-Wall -Wextra -std=c++23 -I.
OUT=main

_DEPS=genivf.hpp
DEPS = $(patsubst %, . %,$(_DEPS))

_OBJ=genivf.o
OBJ = $(patsubst %,$(BUILDDIR)/%,$(_OBJ))

$(BUILDDIR)/%.o: %.cpp $(DEPS)
	$(CC) -c  -o $@ $< $(CFLAGS)

genivf: $(OBJ)
	$(CC) main.cpp -o $@ $^ $(CFLAGS)


__TEST_DEPS=doctest.h
TEST_DEPS = $(patsubst %,$(__TEST_DEPS))

$(BUILDDIR)/%.o: %.h $(__TEST_DEPS)
	$(CC) -c  -o $@ $< $(CFLAGS)

test: $(OBJ)
	$(CC) point.test.cpp -o $@ $^ $(TEST_CFLAGS)

# clang++ -c -Wall -Werror -Wextra -std=c++23 -o build/genivf.o genivf.cpp -I.
