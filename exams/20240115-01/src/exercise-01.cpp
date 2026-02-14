#include "Heat.hpp"

using std::cos, std::sin;

// Main function.
int
main(int argc, char *argv[])
{
  constexpr unsigned int dim = Heat::dim;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const auto mu = [](const Point<dim> & /*p*/) { return 1.; };
  const auto b	= [](const Point<dim> & /*p*/) {
	Tensor<1, dim> tensor;
	tensor[0] = .1;
	tensor[1] = .2;
	return tensor;
  };
  const auto f  = [](const Point<dim>  &p, const double  &t) {
	double value = 5 * M_PI * sin(2 * M_PI * t);
	value += 2 * cos(2 * M_PI * t);
	value *= M_PI * sin(M_PI * p[0]) * sin(2 * M_PI * p[1]);
	value += M_PI / 10 * sin(2 * M_PI * t) * (
		cos(M_PI * p[0]) * sin(2 * M_PI * p[1]) +
		4 * sin(M_PI * p[0]) * cos(2 * M_PI * p[1])
	);
	return value;
  };

  Heat problem(/*mesh_filename = */ "../mesh/mesh-square-h0.100000.msh",
               /* degree = */ 1,
               /* T = */ 1.0,
               /* theta = */ 0.5, // There is decided the method:
								  // 0.0 -> explicit forward Euler
								  // 0.5 -> Crank-Nicolson
								  // 1.0 -> implicit backward Euler (convergence
								  // only for small time steps, in other wards
								  // it needs a small `delta_t`!)
               /* delta_t = */ 0.05,
               mu,
			   b,
               f);

  problem.run();

  return 0;
}
