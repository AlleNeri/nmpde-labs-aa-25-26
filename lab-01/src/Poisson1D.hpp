#ifndef POISSON_1D_HPP
#define POISSON_1D_HPP

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
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

#include <fstream>
#include <iostream>

using namespace dealii;

/**
 * Class managing the differential problem.
 */
class Poisson1D
{
public:
  // Physical dimension (1D, 2D, 3D)
  // This allows to do "dimensional independent programming" by using it
  // instead of hardcoding the dimension everywhere.
  static constexpr unsigned int dim = 1;

  // Constructor.
  // Many of the objects that are going to use the parameters passed to the
  // constructor are going to be stored as unique_ptr, since their type depends
  // on the parameters (otherwise we would need to initialize them even if we
  // don't know all the information yet).
  Poisson1D(const unsigned int                              &N_el_,
            const unsigned int                              &r_,
            const std::function<double(const Point<dim> &)> &mu_,
            const std::function<double(const Point<dim> &)> &f_)
    : N_el(N_el_)
    , r(r_)
    , mu(mu_)
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

  // Compute the error against a given exact solution.
  double
  compute_error(const VectorTools::NormType &norm_type,
                const Function<dim>         &exact_solution) const;

protected:
  // Number of elements (mesh divisions).
  const unsigned int N_el;

  // Polynomial degree (of the FE space).
  const unsigned int r;

  // Diffusion coefficient.
  std::function<double(const Point<dim> &)> mu;

  // Forcing term (right-hand side).
  std::function<double(const Point<dim> &)> f;

  // Triangulation (the mash, it could also be another type).
  Triangulation<dim> mesh;

  // Finite element space.
  //
  // We use a unique_ptr here so that we can choose the type and degree of the
  // finite elements at runtime (the degree is a constructor parameter).
  //
  // The class FiniteElement<dim> is an abstract class from which all types of
  // finite elements implemented by deal.ii inherit. Using the abstract class
  // makes it very easy to switch between different types of FE space among the
  // many that deal.ii provides.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  //
  // We use a unique_ptr here so that we can choose the type and order of the
  // quadrature formula at runtime (the order is a constructor parameter).
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  // This creates a "bridge" between the mesh and the finite element.
  // In some sense, it represents the V_h space.
  DoFHandler<dim> dof_handler;

  // Sparsity pattern of the stiffness matrix.
  SparsityPattern sparsity_pattern;

  // System stiffness matrix (A).
  SparseMatrix<double> system_matrix;

  // System right-hand side (f).
  Vector<double> system_rhs;

  // System solution (u).
  Vector<double> solution;
};

#endif
