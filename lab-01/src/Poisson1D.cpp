#include "Poisson1D.hpp"

void
Poisson1D::setup()
{
  std::cout << "===============================================" << std::endl;

  // Create the mesh.
  {
    std::cout << "Initializing the mesh" << std::endl;
	// The function is called like so because an hyper-cube is the multi-
	// dimensional generalization of a line segment (1D), square (2D), cube (3D)
    GridGenerator::subdivided_hyper_cube(mesh, N_el,
			0.0, 1.0, // Boundaries
			true); // Colorize flag (needed to distinguish among boundaries,
				   // with labels)
    std::cout << "  Number of elements = " << mesh.n_active_cells() // It should
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
    std::cout << "  Mesh saved to " << mesh_file_name << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    std::cout << "Initializing the finite element space" << std::endl;

    // Finite elements in one dimension are obtained with the FE_Q or
    // FE_SimplexP classes (the former is meant for hexahedral elements, the
    // latter for tetrahedra, but they are equivalent in 1D). We use FE_SimplexP
    // here for consistency with the next labs.
    fe = std::make_unique<FE_SimplexP<dim>>(r);

	// `fe->degree` is equivalent to `this.r`
    std::cout << "  Degree                     = " << fe->degree << std::endl;
	// `fe->dofs_per_cell` is the number of nodes per element; in a
	// triangulation of a 2D domain with degree 1, this would be 3 (the triangle
	// vertices)
    std::cout << "  DoFs per cell              = " << fe->dofs_per_cell
              << std::endl;

    // Construct the quadrature formula of the appropriate degree of exactness.
	// A quadrature is an approximation of an integral through a weighted sum
	// of function values at specific points (the quadrature points).
    // This formula integrates exactly the mass matrix terms (i.e. products of
    // basis functions).
    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    std::cout << "  Quadrature points per cell = " << quadrature->size()
              << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    std::cout << "Initializing the DoF handler" << std::endl;

    // Initialize the DoF handler with the mesh we constructed.
    dof_handler.reinit(mesh);

    // "Distribute" the degrees of freedom. For a given finite element space,
    // initializes info on the control variables (how many they are, where
    // they are collocated, their "global indices", ...).
    dof_handler.distribute_dofs(*fe);

	// `dof_handler.n_dofs()` is the total number of DoFs in the mesh, i.e. what
	// we called "N_h" in the theory (it will be the dimension of the stiffness
	// matrix, of the right-hand side vector, and of the solution vector).
    std::cout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    std::cout << "Initializing the linear system" << std::endl;

    // We first initialize a "sparsity pattern", i.e. a data structure that
    // indicates which entries of the matrix are zero and which are different
    // from zero. To do so, we construct first a DynamicSparsityPattern (a
    // sparsity pattern stored in a memory- and access-inefficient way, but
    // fast to write) and then convert it to a SparsityPattern (which is more
    // efficient, but cannot be modified).
    std::cout << "  Initializing the sparsity pattern" << std::endl;
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    // Then, we use the sparsity pattern to initialize the system matrix
    std::cout << "  Initializing the system matrix" << std::endl;
    system_matrix.reinit(sparsity_pattern);

    // Finally, we initialize the right-hand side and solution vectors.
    std::cout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(dof_handler.n_dofs());
    std::cout << "  Initializing the solution vector" << std::endl;
    solution.reinit(dof_handler.n_dofs());
  }
}

void
Poisson1D::assemble()
{
  std::cout << "===============================================" << std::endl;

  std::cout << "  Assembling the linear system" << std::endl;

  // Number of local DoFs for each element. This is the same as n_loc in the
  // theory.
  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  // Number of quadrature points for each element.
  const unsigned int n_q = quadrature->size();

  // FEValues instance. This object allows to compute basis functions, their
  // derivatives, the reference-to-current element mapping and its
  // derivatives on all quadrature points of all elements (It's an utility class
  // these things could be done also by hand).
  // It iterates over the mesh elements and evaluates the function and the
  // derivative quadrature points.
  FEValues<dim> fe_values(
    *fe,
    *quadrature,
    // Here we specify what quantities we need FEValues to compute on
    // quadrature points, in order to avoid unuseful computation. For our test,
	// we need:
    // - the values of shape functions (update_values);
    // - the derivative of shape functions (update_gradients);
    // - the position of quadrature points (update_quadrature_points);
    // - the quadrature weights (update_JxW_values).
    update_values | update_gradients | update_quadrature_points |
      update_JxW_values);

  // Local matrix and right-hand side vector. We will overwrite them for
  // each element within the loop.
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);	// dense matrix
  Vector<double>     cell_rhs(dofs_per_cell);

  // We will use this vector to store the global indices of the DoFs of the
  // current element within the loop.
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  // Reset the global matrix and vector, just in case.
  system_matrix = 0.0;
  system_rhs    = 0.0;

  // Loop over the mesh elements.
  for (const auto &cell : dof_handler.active_cell_iterators()) {
      // Reinitialize the FEValues object on current element. This
      // precomputes all the quantities we requested when constructing
      // FEValues (see the update_* flags above) for all quadrature nodes of
      // the current cell.
      fe_values.reinit(cell);

      // We reset the cell matrix and vector (discarding any leftovers from
      // previous element).
      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      for (unsigned int q = 0; q < n_q; ++q) {
          // Here we assemble the local contribution for current cell and
          // current quadrature point, filling the local matrix and vector.

		  // We precompute these values once since they can involve costly
		  // computations (actually in our example `mu` is going to be a
		  // constant function, but in general we don't know it a priori, also
		  // because it's a constructor argument) and they are used many times
		  // later on.
          const double mu_loc = mu(fe_values.quadrature_point(q));
          const double f_loc  = f(fe_values.quadrature_point(q));

          // Here we iterate over *local* DoF indices.
          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
              for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                  cell_matrix(i, j) += mu_loc *
                                       fe_values.shape_grad(i, q) *
                                       fe_values.shape_grad(j, q) *
                                       fe_values.JxW(q);
				  // `JxW` makes it happen the reference mesh element trick.
				  // It returns a precomputed value (cached), that's the reason
				  // why it doesn't affect much the performance to precompute it
				  // once outside this inner loop and use it here and in the rhs
				  // assembly below (as we did for `mu_loc` and `f_loc`).
              }

              cell_rhs(i) += f_loc *
                             fe_values.shape_value(i, q) *
                             fe_values.JxW(q);
		  }
	  }

      // At this point the local matrix and vector are constructed: we need
      // to sum them into the global matrix and vector. To this end, we need
      // to retrieve the global indices of the DoFs of current cell.
      cell->get_dof_indices(dof_indices);

      // Then, we add the local matrix and vector into the corresponding
      // positions of the global matrix and vector.
      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
    }

  // Boundary conditions.
  //
  // So far we assembled the matrix as if there were no Dirichlet conditions.
  // Now we want to replace the rows associated to nodes on which Dirichlet
  // conditions are applied with equations like u_i = b_i. This isn't the most
  // efficient way to enforce these conditions, but in this kind of problem, the
  // number of boundary equations are usually much smaller w.r.t. the number of
  // total equations, so this is not a big deal (even though very efficient
  // programs use more advanced techniques to avoid unuseful computations on
  // Dirichlet nodes).
  {
    // We construct a map that stores, for each DoF corresponding to a Dirichlet
    // condition, the corresponding value. E.g., if the Dirichlet condition is
    // u_i = b_i, the map will contain the pair (i, b_i).
    std::map<types::global_dof_index, double> boundary_values;

    // This object represents our boundary data as a real-valued function (that
    // always evaluates to zero). Other functions may require to implement a
    // custom class derived from dealii::Function<dim>.
    Functions::ZeroFunction<dim> bc_function;

    // Then, we build a map that, for each boundary tag (portion of boundary),
	// stores a pointer to the corresponding boundary function.
	// It has a more intuitive explanation in 2D/3D, where the boundary
	// conditions may be defined on portions of the boundary instead of on a
	// single point (as in 1D). But for a generalized code, we need to use this
	// approach also in 1D.
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    boundary_functions[0] = &bc_function;
    boundary_functions[1] = &bc_function;

	// `boundary_functions` containes the boundary defined on the whole domain
	// (on the original problem); now we need to apply them to each mash node.
    // interpolate_boundary_values fills the boundary_values map.
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    // Finally, we modify the linear system to apply the boundary conditions.
    // This replaces the equations for the boundary DoFs with the corresponding
    // u_i = 0 equations.
    MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, solution, system_rhs, true);
  }
}

void
Poisson1D::solve()
{
  std::cout << "===============================================" << std::endl;

  // Here we specify the maximum number of iterations of the iterative solver,
  // and its absolute and relative tolerances.
  ReductionControl solver_control(/* maxiter = */ 1000,
                                  /* tolerance = */ 1.0e-16, // Abs. tolerance
                                  /* reduce = */ 1.0e-6); // Relative tolerance
  // The relative tolerance is called "reduction" and it's normalized w.r.t. the
  // norm of the rhs.

  // Since the system matrix is symmetric and positive definite, we solve the
  // system using the conjugate gradient method.
  SolverCG<Vector<double>> solver(solver_control);

  std::cout << "  Solving the linear system" << std::endl;
  // We use the identity preconditioner for now.
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
  std::cout << "  " << solver_control.last_step() << " CG iterations"
            << std::endl;
}

void
Poisson1D::output() const
{
  std::cout << "===============================================" << std::endl;

  // The DataOut class manages writing the results to a file.
  DataOut<dim> data_out;

  // It can write multiple variables (defined on the same mesh) to a single
  // file. Each of them can be added by calling add_data_vector, passing the
  // associated DoFHandler and a name.
  data_out.add_data_vector(dof_handler, solution, "solution");

  // Once all vectors have been inserted, call build_patches to finalize the
  // DataOut object, preparing it for writing to file.
  data_out.build_patches();

  // Then, use one of the many write_* methods to write the file in an
  // appropriate format.
  const std::string output_file_name =
    "output-" + std::to_string(N_el) + ".vtk";
  std::ofstream output_file(output_file_name);
  data_out.write_vtk(output_file);

  std::cout << "Output written to " << output_file_name << std::endl;

  std::cout << "===============================================" << std::endl;
}

double
Poisson1D::compute_error(const VectorTools::NormType &norm_type,
                         const Function<dim>         &exact_solution) const
{
  // The error is an integral, and we approximate that integral using a
  // quadrature formula. To make sure we are accurate enough, we use a
  // quadrature formula with one node more than what we used in assembly. This
  // is done because, otherwise, it's possible that the error comes out to be
  // smaller than the one one in a real world scenario. The additional
  // quadrature point adds a "displacement" which prevents this issue.
  const QGaussSimplex<dim> quadrature_error(r + 2);

  // First we compute the norm of the error on each element.
  Vector<double> error_per_cell(mesh.n_active_cells());
  VectorTools::integrate_difference(dof_handler,  // Mesh
                                    solution, // Numerical solution
                                    exact_solution,	// Exact solution
                                    error_per_cell, // Output vector
                                    quadrature_error, // Quadrature
                                    norm_type);	// Norm type

  // Then, we add out all the cells.
  const double error =
    VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}
