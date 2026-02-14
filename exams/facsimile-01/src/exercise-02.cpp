#include <deal.II/base/convergence_table.h>

#include <iostream>
#include <utility>

#include "DiffusionReaction.hpp"

using namespace std;
using namespace DiffusionReactionNamespace;

constexpr unsigned int dim = DiffusionReaction::dim;

// Exact solution.
class ExactSolution : public Function<dim> {
public:
	// Constructor.
	ExactSolution() /*: Function<dim>(1)*/ {}

	// Evaluate.
	virtual double value(const Point<dim> &p,
					  const unsigned int /*component*/ = 0) const override {
		return sin(M_PI / 2. * p[0]) * p[1];
	}

	// Gradient evaluation.
	virtual Tensor<1, dim> gradient(const Point<dim> &p,
						const unsigned int /*component*/ = 0) const override {
		Tensor<1, dim> result;
		result[0] = (M_PI / 2.) * cos(M_PI / 2. * p[0]) * p[1];
		result[1] = sin(M_PI / 2. * p[0]);
		return result;
	}
};

#define CONST_FUNCTION(x) [](const Point<dim> &/*p*/) { return x; }

int main(int /*argc*/, char* /*argv*/[]) {
	ConvergenceTable table;

	const vector<pair<string, double>> meshes = {
		make_pair("../mesh/mesh-square-h0.100000.msh", 0.1),
		make_pair("../mesh/mesh-square-h0.050000.msh", 0.05),
		make_pair("../mesh/mesh-square-h0.025000.msh", 0.025),
		make_pair("../mesh/mesh-square-h0.012500.msh", 0.0125)
	};
	const unsigned int r = 2;
	const auto mu = CONST_FUNCTION(1.);
	const auto sigma = CONST_FUNCTION(1.);
	const auto f = [](const Point<dim> &p) {
		// Compute: (1 + pi^2 / 4) * sin(pi / 2 * x) * y
		return (1. + M_PI*M_PI / 4.) * sin(M_PI / 2. * p[0]) * p[1];
	};

	const ExactSolution exact_solution;

	ofstream convergence_file("convergence.csv");
	convergence_file << "h,eL2,eH1" << endl;

	for (const auto &[mesh_file_name, h] : meshes) {
		DiffusionReaction problem(mesh_file_name, r, mu, sigma, f);

		problem.setup();
		problem.assemble();
		problem.solve();
		problem.output();

		// Compute errors.
		const double error_L2 = problem.compute_error(VectorTools::L2_norm,
												exact_solution);
		const double error_H1 = problem.compute_error(VectorTools::H1_norm,
												exact_solution);

		// Add errors to the convergence table.
		table.add_value("h", h);
		table.add_value("L2", error_L2);
		table.add_value("H1", error_H1);

		// add error value to the CSV file.
		convergence_file << h << "," << error_L2 << "," << error_H1 << endl;
	}

	// Compute the convergence rates.
	table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
	// Format the table.
	table.set_scientific("L2", true);
	table.set_scientific("H1", true);

	table.write_text(cout);

	return 0;
}
