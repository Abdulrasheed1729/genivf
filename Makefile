BUILDDIR    = build
CC          = clang++
CFLAGS      = -Wall -Werror -Wextra -std=c++23 -Iinclude -O3 -march=native -ffast-math -flto
# -Itests resolves doctest.h now that it lives in tests/.
TEST_CFLAGS = -Wall -Wextra -std=c++23 -Iinclude -Itests -O3 -march=native -ffast-math -flto

OBJ = $(BUILDDIR)/genivf.o $(BUILDDIR)/flat.o $(BUILDDIR)/seq.o $(BUILDDIR)/io.o

.PHONY: all clean

all: genivf point_test genivf_test io_test seq_test flat_test utils_test simple build_flat_index measure_accuracy

# Create the build directory if it does not exist.
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Compile the library objects. Headers listed explicitly so Make re-builds when
# they change.
$(BUILDDIR)/genivf.o: src/genivf.cpp include/genivf.hpp include/utils.hpp | $(BUILDDIR)
	$(CC) -c -o $@ $< $(CFLAGS)

$(BUILDDIR)/seq.o: src/seq.cpp include/seq.hpp | $(BUILDDIR)
	$(CC) -c -o $@ $< $(CFLAGS)

$(BUILDDIR)/flat.o: src/flat.cpp include/flat.hpp | $(BUILDDIR)
	$(CC) -c -o $@ $< $(CFLAGS)

$(BUILDDIR)/io.o: src/io.cpp include/io.hpp include/genivf.hpp include/flat.hpp | $(BUILDDIR)
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

flat_test: tests/flat.test.cpp $(OBJ)
	$(CC) tests/flat.test.cpp -o $@ $(OBJ) $(TEST_CFLAGS)

utils_test: tests/utils.test.cpp $(OBJ)
	$(CC) tests/utils.test.cpp -o $@ $(OBJ) $(TEST_CFLAGS)

build_flat_index: examples/build_flat_index.cpp $(OBJ)
	$(CC) examples/build_flat_index.cpp -o $@ $(OBJ) $(CFLAGS)

measure_accuracy: examples/measure_accuracy.cpp $(OBJ)
	$(CC) examples/measure_accuracy.cpp -o $@ $(OBJ) $(CFLAGS)

clean:
	rm -rf $(BUILDDIR) genivf point_test genivf_test io_test seq_test flat_test utils_test simple build_flat_index measure_accuracy


