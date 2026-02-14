#include <iostream>

#include "DiffusionReaction.hpp"

#define CONST_FUNCTION(x) [](const Point<dim> &/*p*/) { return x; }

using namespace std;
using namespace DiffusionReactionNamespace;

int main(int argc, char* argv[]) {
	if(argc != 2) {
		cerr << "Usage: " << argv[0] << " <mesh_file_name>" << endl;
		return 1;
	}

	constexpr unsigned int dim = DiffusionReaction::dim;

	const string mesh_file_name = argv[1];
	const unsigned int r = 2;

	const auto mu = CONST_FUNCTION(1.);
	const auto sigma = CONST_FUNCTION(1.);
	const auto f = [](const Point<dim> &p) {
		// Compute: (1 + pi^2 / 4) * sin(pi / 2 * x) * y
		return (1. + M_PI*M_PI / 4.) * sin(M_PI / 2. * p[0]) * p[1];
	};

	DiffusionReaction problem(mesh_file_name, r, mu, sigma, f);

	problem.setup();
	problem.assemble();
	problem.solve();
	problem.output();

	return 0;
}
