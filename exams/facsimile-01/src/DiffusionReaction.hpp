#ifndef DIFFUSIONREACTION_HPP
#define DIFFUSIONREACTION_HPP

#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

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
using std::function;
using std::string;
using std::abs;
using std::unique_ptr;

namespace DiffusionReactionNamespace {
class DiffusionReaction {
public:
	static constexpr unsigned int dim = 2;

	// Dirichlet boundary function.
	// This function represents the function g which enforces the Dirichlet BCs
	// for this specific problem. This is implemented as a dealii::Function<dim>,
	// instead of e.g. a lambda function, because this allows to use dealii
	// boundary utilities directly.
	class FunctionG : public Function<dim>
	{
	public:
		// Constructor.
		FunctionG() = default;

		// Evaluation.
		virtual double
		value(const Point<dim> &p,
			const unsigned int /*component*/ = 0) const override
		{
			// This function simply performs:
			// g(x,y) = 0 if x = 0 or y = 0
			if(p[0] == 0. || p[1] == 0.)
				return 0.;
			// g(x,y) = sin(pi / 2 * x) if y = 1
			else if(p[1] == 1.)
				return sin((numbers::PI / 2.) * p[0]);
			else  // It should not reach this point.
				return 0.;
		}
	};


	DiffusionReaction(const string &mesh_file_name_,
					const unsigned int &r_,
					const function<double(const Point<dim> &)> &mu_,
					const function<double(const Point<dim> &)> &sigma_,
					const function<double(const Point<dim> &)> &f_)
		: mesh_file_name(mesh_file_name_)
		, r(r_)
		, mu(mu_)
		, sigma(sigma_)
		, f(f_)
	{}

	// Initialization.
	void setup();

	// System assembly.
	void assemble();

	// System solution.
	void solve();

	// Output.
	void output() const;

	// Compute the error against a given exact solution.
	double compute_error(const VectorTools::NormType &norm_type,
					  const Function<dim> &exact_solution) const;

protected:
	// Name of the file containing the mesh.
	const string mesh_file_name;

	// Polynomial degree.
	const unsigned int r;

	// Diffusion coefficient.
	function<double(const Point<dim> &)> mu;

	// Reaction coefficient.
	function<double(const Point<dim> &)> sigma;

	// Forcing term.
	function<double(const Point<dim> &)> f;

	// Triangulation.
	Triangulation<dim> mesh;

	// Finite element space.
	unique_ptr<FiniteElement<dim>> fe;

	// Quadrature formula.
	unique_ptr<Quadrature<dim>> quadrature;

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
}

#endif // DIFFUSIONREACTION_HPP
