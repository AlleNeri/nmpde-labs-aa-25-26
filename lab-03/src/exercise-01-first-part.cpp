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

  const auto mu	   = [](const Point<dim> &p) {
	if(p[0] < 0.5)
	  return 100.;
	else
	  return 1.;
  };
  const auto sigma = CONST_FUNCTION(1.);
  const auto f	   = CONST_FUNCTION(1.);

  DiffusionReaction problem(mesh_file_name, r, mu, sigma, f);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();

  return 0;
}
