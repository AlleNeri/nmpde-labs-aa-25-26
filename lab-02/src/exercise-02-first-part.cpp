#include <iostream>

#include "my-DiffusionReaction.hpp"

#define CONST_FUNCTION(x) [](const Point<dim> &/*p*/) { return x; }

using namespace std;
using namespace myDiffusionReaction;

int main(int argc, char *argv[]) {
  if(argc != 2) {
	cerr << "Usage: " << argv[0] << " <mesh_file_name>" << endl;
	return 1;
  }

  constexpr unsigned int dim = DiffusionReaction::dim;

  const string mesh_file_name = argv[1];
  const unsigned int r = 1;

  const auto mu	   = CONST_FUNCTION(1.0);
  const auto sigma = CONST_FUNCTION(1.0);
  const auto f	   = [](const Point<dim> &p) {
	return (20. * M_PI * M_PI + 1.) * sin(2. * M_PI * p[0])
									* sin(4. * M_PI * p[1]);
  };

  DiffusionReaction problem(mesh_file_name, r, mu, sigma, f);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();

  return 0;
}
