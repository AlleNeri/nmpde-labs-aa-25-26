#include "Poisson2D.hpp"

void
Poisson2D::setup()
{
  std::cout << "===============================================" << std::endl;

  // Create the mesh.
  {
    std::cout << "Initializing the mesh" << std::endl;

    // Read the mesh from file.
    GridIn<dim> grid_in;  // This object is a mesh reader.
    grid_in.attach_triangulation(mesh);	// With this, we are connecting the mesh
										// object with the mesh reader.

    std::ifstream mesh_file(mesh_file_name);  // The actual file stream.
    grid_in.read_msh(mesh_file);  // This object supports various mesh formats
								  // (see the documentation); this method reads
								  // the Gmsh format.

    std::cout << "  Number of elements = " << mesh.n_active_cells()
              << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    std::cout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);

    std::cout << "  Degree                     = " << fe->degree << std::endl;
    std::cout << "  DoFs per cell              = " << fe->dofs_per_cell
              << std::endl;

    quadrature          = std::make_unique<QGaussSimplex<dim>>(r + 1);
    quadrature_boundary = std::make_unique<QGaussSimplex<dim - 1>>(r + 1);

    std::cout << "  Quadrature points per cell = " << quadrature->size()
              << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    std::cout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    std::cout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    std::cout << "Initializing the linear system" << std::endl;

    std::cout << "  Initializing the sparsity pattern" << std::endl;
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    std::cout << "  Initializing the system matrix" << std::endl;
    system_matrix.reinit(sparsity_pattern);

    std::cout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(dof_handler.n_dofs());
    std::cout << "  Initializing the solution vector" << std::endl;
    solution.reinit(dof_handler.n_dofs());
  }
}

void
Poisson2D::assemble()
{
  std::cout << "===============================================" << std::endl;

  std::cout << "  Assembling the linear system" << std::endl;

  // Number of local DoFs for each element.
  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  // Number of quadrature points for each element.
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  // Since we need to compute integrals on the boundary for Neumann conditions,
  // we also need a FEValues object to compute quantities on boundary edges
  // (faces).
  // This is similar to the previous object (indeed, its initialization is
  // almost the same), it wasn't needed in the 1D case because the BCs weren't
  // involving integrals (we only had Dirichlet conditions there).
  FEFaceValues<dim> fe_values_boundary(*fe,
                                       *quadrature_boundary,
                                       update_values |
                                         update_quadrature_points |
                                         update_JxW_values);
									// Since the BCs only involve the values of
									// the function (and not its derivative), we
									// can omit the `update_gradients` flag in
									// this case. It saves some computation.

  // Local matrix and vector.
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  // Reset the global matrix and vector, just in case.
  system_matrix = 0.0;
  system_rhs    = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const double mu_loc = mu(fe_values.quadrature_point(q));
          const double f_loc  = f(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
				  // Since we are working in a 2D space, the product between the
				  // two gradients in the following lines, is a dot product.
				  // The `*` operator is appropriately overloaded in deal.II to
				  // handle this case (the explicit function which computes this
				  // operation is `scalar_product`).
                  cell_matrix(i, j) += mu_loc *
                                       fe_values.shape_grad(i, q) *
                                       fe_values.shape_grad(j, q) *
                                       fe_values.JxW(q);
                }

              cell_rhs(i) += f_loc *
                             fe_values.shape_value(i, q) *
                             fe_values.JxW(q);
            }
        }

	  // Neumann boundary conditions.
	  //
      // If the cell (in our 2D example, a triangle) is adjacent to the boundary
	  // ...
      if (cell->at_boundary())
        {
          // ... we loop over its edges (referred to as faces in the deal.II
          // jargon). Since we are in 2D, the faces in our case are edges of a
		  // line segment.
          for (unsigned int face_number = 0; face_number < cell->n_faces();
               ++face_number)
            {
              // If current face lies on the boundary (first condition), and its
			  // boundary ID (or tag) is that of one of the Neumann boundaries,
			  // we assemble the boundary integral(second and third conditions).
			  // In this way we are imposing Neumann BCs only, not the Dirichlet
			  // ones.
			  // NOTE: the boundary IDs used here (2 and 3) are problem-specific
			  // information; they depend on how the mesh was created. In this
			  // case we know that the mesh we pass to the program are correctly
			  // tagged. Nonetheless, they shouldn't be hard-coded in this way,
			  // but rather passed to the class from outside.
              if (cell->face(face_number)->at_boundary() &&
                  (cell->face(face_number)->boundary_id() == 2 ||
                   cell->face(face_number)->boundary_id() == 3))
                {
				  // In here the structure is similar to what we had before for
				  // the "regular" rhs assembly, but instead of using the object
				  // which represents quantities in the inner triangles, we use
				  // the one for boundary faces. This reflects on the code below
				  // in some small differences.
                  fe_values_boundary.reinit(cell, face_number);

                  for (unsigned int q = 0; q < quadrature_boundary->size(); ++q)
                    {
					  // The integral we are going to compute involves also the
					  // function h, which is a class member and it's evaluated
					  // at the quadrature point. As we did before, since the
					  // value is going to be used multiple times, we store it
					  // in a local variable.
					  const double h_loc =
						  h(fe_values_boundary.quadrature_point(q));

                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
						// We add to the already computed value the contribution
						// of the integral coming from the Neumann boundary.
                        cell_rhs(i) +=
                          h_loc *
                          fe_values_boundary.shape_value(i, q) *
                          fe_values_boundary.JxW(q);
                    }
                }
            }
        }

      cell->get_dof_indices(dof_indices);

      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
    }

  // Dirichlet boundary conditions.
  {
    std::map<types::global_dof_index, double> boundary_values;
    FunctionG                                 bc_function;

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
	// In there, as discussed before, we are referring to the boundaries by
	// using their IDs/tags (0 and 1); but this hard-coding is not ideal.
    boundary_functions[0] = &bc_function;
    boundary_functions[1] = &bc_function;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, solution, system_rhs, true);
  }
}

void
Poisson2D::solve()
{
  std::cout << "===============================================" << std::endl;

  ReductionControl solver_control(/* maxiter = */ 1000,
                                  /* tolerance = */ 1.0e-16,
                                  /* reduce = */ 1.0e-6);

  SolverCG<Vector<double>> solver(solver_control);

  std::cout << "  Solving the linear system" << std::endl;
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
  std::cout << "  " << solver_control.last_step() << " CG iterations"
            << std::endl;
}

void
Poisson2D::output() const
{
  std::cout << "===============================================" << std::endl;

  DataOut<dim> data_out;

  data_out.add_data_vector(dof_handler, solution, "solution");
  data_out.build_patches();

  // Use std::filesystem to construct the output file name based on the
  // mesh file name.
  const std::filesystem::path mesh_path(mesh_file_name);
  const std::string           output_file_name =
    "output-" + mesh_path.stem().string() + ".vtk";	// `.stem()` removes the
													// file extension.
  std::ofstream output_file(output_file_name);
  data_out.write_vtk(output_file);

  std::cout << "Output written to " << output_file_name << std::endl;

  std::cout << "===============================================" << std::endl;
}
