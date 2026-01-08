#include "Poisson2D.hpp"

// Main function.
int
main(int /*argc*/, char * /*argv*/[])
{
  // The fact that there should be two types of communication (the one among
  // processes and the one among subdomains) makes this type of problem much
  // more complex to implement in a parallel way; for this example, we won't use
  // MPI.
  // Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  // The following code is the actual implementation of the Dirichlet-Neumann
  // algorithm for the solution of the Poisson equation in two subdomains.

  // Initialize the two subdomain problems.
  Poisson2D problem_0(0);
  Poisson2D problem_1(1);

  problem_0.setup();
  problem_1.setup();

  std::cout << "Setup completed" << std::endl;

  // Iterative solution of the coupled problems.

  // Define the stopping criteria.
  const double       tolerance_increment = 1e-4;
  const unsigned int n_max_iter          = 100;

  // Initialize the sequences of values to monitor convergence.
  double       solution_increment_norm = tolerance_increment + 1;
  unsigned int n_iter                  = 0;

  // Relaxation coefficient (1 = no relaxation).
  const double lambda = 0.25;

  // Solve the coupled problems in the proper interleaved manner.
  while (n_iter < n_max_iter && solution_increment_norm > tolerance_increment)
    {
	  // Solution of problem 0 will employ the solution at the previous
	  // iteration of problem 1.
      auto solution_1_increment = problem_1.get_solution();

	  // Solve problem 0 with Dirichlet BC from problem 1 (at the previous
	  // iteration).
      problem_0.assemble();
      problem_0.apply_interface_dirichlet(problem_1);
      problem_0.solve();

	  // Solve problem 1 with Neumann BC from problem 0 (at the current
	  // iteration).
      problem_1.assemble();
      problem_1.apply_interface_neumann(problem_0);
      problem_1.solve();

	  // Relaxation step for the solution of problem 1 (needed for a better
	  // convergence of the overall algorithm).
      problem_1.apply_relaxation(solution_1_increment, lambda);

	  // Compute the solution increment for convergence monitoring.
      solution_1_increment -= problem_1.get_solution();
      solution_increment_norm = solution_1_increment.l2_norm();

      std::cout << "iteration " << n_iter
                << " - solution increment = " << solution_increment_norm
                << std::endl;

	  // Output at every iteration for teaching purposes.
      problem_0.output(n_iter);
      problem_1.output(n_iter);

      ++n_iter;
    }

  return 0;
}
