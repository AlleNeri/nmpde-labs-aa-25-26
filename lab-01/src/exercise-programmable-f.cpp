#include <iostream>

#include "Poisson1D.hpp"

// Main function.
int
main(int argc, char * argv[])
{
  if (argc != 4) {
	std::cerr << "Usage: " << argv[0] << " start_force end_force force"
		<< std::endl;
	return 1;
  }

  double start_force, end_force, force;

  try {
	start_force = std::stod(argv[1]);
	end_force   = std::stod(argv[2]);
	force       = std::stod(argv[3]);
  } catch (const std::invalid_argument &e) {
	std::cerr << "Invalid arguments. start_force, end_force and force must be "
			  << "numbers." << std::endl;
	return 1;
  }

  if (start_force >= end_force)
	  std::swap(start_force, end_force);

  if (start_force < 0 || end_force > 1 || start_force >= end_force) {
	std::cerr << "Invalid force interval. Must satisfy 0 <= start_force < "
		<< "end_force <= 1." << std::endl;
	return 1;
  }

  constexpr unsigned int dim = Poisson1D::dim;

  const unsigned int N_el = 40;
  const unsigned int r    = 2;
  // `Point` object is a sort of vector representing a point in the domain.
  const auto mu = [](const Point<dim> &/*p*/) { return 1.0; };
  const auto f  = [start_force, end_force, force](const Point<dim> &p) {
    if (p[0] <= start_force || p[0] > end_force)
      return 0.0;
    else
      return force;
  };

  Poisson1D problem(N_el, r, mu, f);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();

  return 0;
}
