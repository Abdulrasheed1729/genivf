BUILDDIR    = build
CC          = clang++
CFLAGS      = -Wall -Werror -Wextra -std=c++23 -Iinclude
# -Itests resolves doctest.h now that it lives in tests/.
TEST_CFLAGS = -Wall -Wextra -std=c++23 -Iinclude -Itests

OBJ = $(BUILDDIR)/genivf.o

.PHONY: all clean

all: genivf point_test genivf_test io_test

# Create the build directory if it does not exist.
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Compile the library object. Headers listed explicitly so Make re-builds when
# they change.
$(BUILDDIR)/genivf.o: src/genivf.cpp include/genivf.hpp include/utils.hpp | $(BUILDDIR)
	$(CC) -c -o $@ $< $(CFLAGS)

genivf: main.cpp $(OBJ)
	$(CC) main.cpp -o $@ $(OBJ) $(CFLAGS)

point_test: tests/point.test.cpp $(OBJ)
	$(CC) tests/point.test.cpp -o $@ $(OBJ) $(TEST_CFLAGS)

genivf_test: tests/genivf.test.cpp $(OBJ)
	$(CC) tests/genivf.test.cpp -o $@ $(OBJ) $(TEST_CFLAGS)

io_test: tests/io.test.cpp $(OBJ)
	$(CC) tests/io.test.cpp -o $@ $(OBJ) $(TEST_CFLAGS)

clean:
	rm -rf $(BUILDDIR) genivf point_test genivf_test io_test
