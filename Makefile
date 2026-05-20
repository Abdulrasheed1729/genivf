BUILDDIR    = build
CC          = clang++
CFLAGS      = -Wall -Werror -Wextra -std=c++23 -Iinclude -O3 -march=native -ffast-math -flto
# -Itests resolves doctest.h now that it lives in tests/.
TEST_CFLAGS = -Wall -Wextra -std=c++23 -Iinclude -Itests -O3 -march=native -ffast-math -flto

OBJ = $(BUILDDIR)/genivf.o $(BUILDDIR)/seq.o

.PHONY: all clean

all: genivf point_test genivf_test io_test seq_test simple

# Create the build directory if it does not exist.
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Compile the library objects. Headers listed explicitly so Make re-builds when
# they change.
$(BUILDDIR)/genivf.o: src/genivf.cpp include/genivf.hpp include/utils.hpp | $(BUILDDIR)
	$(CC) -c -o $@ $< $(CFLAGS)

$(BUILDDIR)/seq.o: src/seq.cpp include/seq.hpp | $(BUILDDIR)
	$(CC) -c -o $@ $< $(CFLAGS)


genivf: main.cpp $(OBJ)
	$(CC) main.cpp -o $@ $(OBJ) $(CFLAGS)

point_test: tests/point.test.cpp $(OBJ)
	$(CC) tests/point.test.cpp -o $@ $(OBJ) $(TEST_CFLAGS)

genivf_test: tests/genivf.test.cpp $(OBJ)
	$(CC) tests/genivf.test.cpp -o $@ $(OBJ) $(TEST_CFLAGS)

io_test: tests/io.test.cpp $(OBJ)
	$(CC) tests/io.test.cpp -o $@ $(OBJ) $(TEST_CFLAGS)

seq_test: tests/seq.test.cpp $(OBJ)
	$(CC) tests/seq.test.cpp -o $@ $(OBJ) $(TEST_CFLAGS)

simple: examples/simple.cpp $(OBJ)
	$(CC) examples/simple.cpp -o $@ $(OBJ) $(CFLAGS)

clean:
	rm -rf $(BUILDDIR) genivf point_test genivf_test io_test seq_test simple


