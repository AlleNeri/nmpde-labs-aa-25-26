#include "Heat.hpp"

void
Heat::setup()
{
  pcout << "===============================================" << std::endl;

  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;
	// The function is called like so because an hyper-cube is the multi-
	// dimensional generalization of a line segment (1D), square (2D), cube (3D)
    GridGenerator::subdivided_hyper_cube(mesh, N_el,
			0.0, 1.0, // Boundaries
			true); // Colorize flag (needed to distinguish among boundaries,
				   // with labels)
    pcout << "  Number of elements = " << mesh.n_active_cells() // It should
																	// be N_el
              << std::endl;

    // Write the mesh to file.
    // Since we generate the mesh internally, we also write it to file for
    // possible inspection by the user. This would not be necessary if we read
    // the mesh from file, as we will do later on.
    const std::string mesh_file_name = "mesh-" + std::to_string(N_el) + ".vtk";
    GridOut           grid_out;
    std::ofstream     grid_out_file(mesh_file_name);
    grid_out.write_vtk(mesh, grid_out_file);
    pcout << "  Mesh saved to " << mesh_file_name << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

    pcout << "  Initializing the sparsity pattern" << std::endl;
    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    pcout << "  Initializing the system matrix" << std::endl;
    system_matrix.reinit(sparsity);

    pcout << "  Initializing vectors" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
	// Notice the double initialization of solution vectors.
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  }
}

void
Heat::assemble()
{
  // Number of local DoFs for each element.
  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  // Number of quadrature points for each element.
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  // Local matrix and vector.
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  // Reset the global matrix and vector, just in case.
  system_matrix = 0.0;
  system_rhs    = 0.0;

  // Due to time discretization also the solution at the previous time step is
  // needed; for that reason why we use the following two vectors.

  // Evaluation of the old solution on quadrature nodes of current cell.
  std::vector<double> solution_old_values(n_q);

  // Evaluation of the gradient of the old solution on quadrature nodes of
  // current cell.
  std::vector<Tensor<1, dim>> solution_old_grads(n_q);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      // Evaluate the old solution and its gradient on quadrature nodes.
      fe_values.get_function_values(solution, solution_old_values);
      fe_values.get_function_gradients(solution, solution_old_grads);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const double k_loc = k(fe_values.quadrature_point(q));

		  // For generality, in our case it's 0
          const double f_old_loc =
            f(fe_values.quadrature_point(q), time - delta_t);
          const double f_new_loc = f(fe_values.quadrature_point(q), time);

		  // Notice: in this simple linear problem the mass and stiffness
		  // matrices M & A are time-independent; however, for that reason we
		  // could compute them only once at the beginning of the simulation and
		  // just recompute the time-dependent forcing term and just change the
		  // right-hand side at each time step.
		  // For a more general problem (e.g., nonlinear, time-dependent
		  // coefficients, etc.) we need to reassemble the whole system at each
		  // time step as done here.
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // Time derivative (M / Delta t).
                  cell_matrix(i, j) += (1.0 / delta_t) *  // 1 / Delta t
                                       (
										 fe_values.shape_value(i, q) *
                                         fe_values.shape_value(j, q) *
                                         fe_values.JxW(q)
									   ); // mass matrix M

                  // Diffusion (theta * A).
                  cell_matrix(i, j) += theta *	// theta coefficient(time disc.)
					(
					  (
						scalar_product(fe_values.shape_grad(i, q),
									   fe_values.shape_grad(j, q)) *
						fe_values.JxW(q)
					  ) + (
						k_loc *
						fe_values.shape_value(j, q) *
						fe_values.shape_grad(i, q)[0] *
						fe_values.JxW(q)
					  )
					);					  // matrix A (computed with quadrature)
                }

              // Time derivative (M / Delta t).
              cell_rhs(i) += (1.0 / delta_t) *	// 1 / Delta t
                             (
							   fe_values.shape_value(i, q) *
							   solution_old_values[q] *
							   fe_values.JxW(q)
							 );	// matrix M at previous time step (quadrature)

              // Diffusion (-(1 - theta)A).
              cell_rhs(i) -= (1.0 - theta) *  // 1 - theta
							 (
							   (
							     scalar_product(fe_values.shape_grad(i, q),
												solution_old_grads[q]) *
							     fe_values.JxW(q)
							   ) + (
							     k_loc *
							     solution_old_values[q] *
							     fe_values.shape_grad(i, q)[0] *
							     fe_values.JxW(q)
							   )
							 );				  // matrix A (with quadrature)
			  // Notice: we don't have this term in this case since tetha = 1.
			  // However, it's included here for generality.

              // Forcing term (theta F^{n+1} + (1 - theta) F^n).
              cell_rhs(i) += (
				  (theta * f_new_loc) +		  // theta F^{n+1}
				  ((1.0 - theta) * f_old_loc) // (1 - theta) F^n
				) * (fe_values.shape_value(i, q) * fe_values.JxW(q)); // quadr.
			  // Notice: we don't have this term in this case since f = 0.
			  // However, it's included here for generality.
            }
        }

      cell->get_dof_indices(dof_indices);

      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  // Dirichlet boundary conditions.
  {
	std::map<types::global_dof_index, double> boundary_values;

	// Homogeneous Dirichlet BCs on one side of the domain.
	Functions::ZeroFunction<dim> zero_function;
	
	// Constant Dirichlet BCs on the other side of the domain.
	Functions::ConstantFunction<dim> one_function(1.);

	// Apply zero Dirichlet BCs on boundary with id 0.
  }
}

void
Heat::solve_linear_system()
{
  TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize(
    system_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  ReductionControl solver_control(/* maxiter = */ 10000,
                                  /* tolerance = */ 1.0e-16,
                                  /* reduce = */ 1.0e-6);

  // SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);

  // Notice: in this passage we put the solution inside the vector
  // `solution_owned` instead of `solution`.
  solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
  pcout << solver_control.last_step() << " CG iterations" << std::endl;
}

void
Heat::output() const
{
  DataOut<dim> data_out;

  data_out.add_data_vector(dof_handler, solution, "solution");

  // Add vector for parallel partition.
  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  const std::string output_file_name = "output-" + std::to_string(N_el)+ ".vtk";

  // Notice: the `index` parameter of this function is set to timestep_number in
  // order to model the time evolution of the problem in the solution output.
  data_out.write_vtu_with_pvtu_record(/* folder = */ "./",
                                      /* basename = */ output_file_name,
                                      /* index = */ timestep_number,
                                      MPI_COMM_WORLD);
}

void
Heat::run()
{
  // Setup initial conditions.
  {
    setup();

	// This function fills the `solution_owned` variables with the interpolation
	// of function u_0 for each degree of freedom; this is the approximation of
	// the continuous function u_0 in our finite element space.
    VectorTools::interpolate(dof_handler, FunctionU0(), solution_owned);
    solution = solution_owned;	// This assignment hides parallel communication.
								// The same appens hereafter in the loop.

	// Initialize the values of time step.
    time            = 0.0;
    timestep_number = 0;

    // Output initial condition.
    output();
  }

  pcout << "===============================================" << std::endl;

  // Time-stepping loop.

  // Morally the condition is just `time < T`; in order to take into account the
  // arithmetic errors due to the repetition of the floating point sum, a
  // tolerance of (Delta t / 2) is added.
  while (time < T - 0.5 * delta_t)
    {
      time += delta_t;
      ++timestep_number;

      pcout << "Timestep " << std::setw(3) << timestep_number
            << ", time = " << std::setw(4) << std::fixed << std::setprecision(2)
            << time << " : ";

      assemble();
      solve_linear_system();

      // Perform parallel communication to update the ghost values of the
      // solution vector.
      solution = solution_owned;

	  // Notice: this call, which will be repeated multiple times, together with
	  // the one in the initialization block; will create a sequence of
	  // solutions which represents the time step evolution.
      output();
    }
}
