#ifndef POISSON_2D_HPP
#define POISSON_2D_HPP

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace dealii;

namespace myDiffusionReaction {
/**
 * Class managing the differential problem.
 */
class DiffusionReaction
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;

  // Constructor.
  DiffusionReaction(const std::string						&mesh_file_name_,
            const unsigned int                              &r_,
            const std::function<double(const Point<dim> &)> &mu_,
			const std::function<double(const Point<dim> &)> &sigma_,
            const std::function<double(const Point<dim> &)> &f_)
    : mesh_file_name(mesh_file_name_)
    , r(r_)
    , mu(mu_)
	, sigma(sigma_)
    , f(f_)
  {}

  // Initialization.
  void
  setup();

  // System assembly.
  void
  assemble();

  // System solution.
  void
  solve();

  // Output.
  void
  output() const;

protected:
  // Name of the file containing the mesh.
  const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Diffusion coefficient.
  std::function<double(const Point<dim> &)> mu;

  // Reaction coefficient.
  std::function<double(const Point<dim> &)> sigma;

  // Forcing term.
  std::function<double(const Point<dim> &)> f;

  // Triangulation.
  Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // Sparsity pattern.
  SparsityPattern sparsity_pattern;

  // System matrix.
  SparseMatrix<double> system_matrix;

  // System right-hand side.
  Vector<double> system_rhs;

  // System solution.
  Vector<double> solution;
};
} // namespace myDiffusionReaction

#endif
