#include "Heat.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  constexpr unsigned int dim = Heat::dim;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const auto k = [](const Point<dim> &/*p*/) { return 25.; };
  const auto f = [](const Point<dim> &/*p*/, const double &/*t*/) {
    return 0.0;
  };

  Heat problem(/* N_el = */ 10,
               /* degree = */ 1,
               /* T = */ 1.0,
               /* theta = */ 1.0, // There is decided the method:
								  // 0.0 -> explicit forward Euler
								  // 0.5 -> Crank-Nicolson
								  // 1.0 -> implicit backward Euler (convergence
								  // only for small time steps, in other wards
								  // it needs a small `delta_t`!)
               /* delta_t = */ 0.1,
               k,
               f);

  problem.run();

  return 0;
}
