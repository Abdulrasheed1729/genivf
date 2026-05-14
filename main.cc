#include "genivf.hpp"
#include <vector>
#include <print>

int
main()
{
	std::vector<genivf::Point> points = {
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
		{10.0, 11.0, 12.0},
	};

	genivf::IndexIVF ivf{2, 3};
	ivf.train(points, 100, 1e-5);
	std::println("Training a small IVF");

  sayHello();
}
