#include <deal.II/base/convergence_table.h>

#include <iostream>

#include "Poisson1D.hpp"

enum function_t { sine, step };

static constexpr unsigned int dim = Poisson1D::dim;

// Exact solution.
class ExactSolution : public Function<dim>
{
private:
  const function_t _function;
public:
  // Constructor.
  ExactSolution(const function_t function): _function(function)
  {}

  // Evaluation.
  virtual double
  value(const Point<dim> &p,
        const unsigned int /*component*/ = 0) const override
  {
	switch (_function) {
	  case sine:	// Points 3 and 4.
		return std::sin(2.0 * M_PI * p[0]);
	  case step: // Point 5.
		if (p[0] < 0.5)
		  return A * p[0];
		else
		  return A * p[0] + 4.0 / 15.0 * std::pow(p[0] - 0.5, 2.5);
	}
  }

  // Gradient evaluation.
  // deal.II requires this method to return a Tensor (not a double), i.e. a
  // dim-dimensional vector. In our case, dim = 1, so that the Tensor will in
  // practice contain a single number. Nonetheless, we need to return an
  // object of type Tensor.
  virtual Tensor<1, dim>
  gradient(const Point<dim> &p,
           const unsigned int /*component*/ = 0) const override
  {
    Tensor<1, dim> result;

	switch (_function) {
	  case sine: // Points 3 and 4.
		result[0] = 2.0 * M_PI * std::cos(2.0 * M_PI * p[0]);
		break;
	  case step: // Point 5.
		if (p[0] < 0.5)
		  result[0] = A;
		else
		  result[0] = A + 2.0 / 3.0 * std::pow(p[0] - 0.5, 1.5);
		break;
	}

    return result;
  }

  static constexpr double A = -4.0 / 15.0 * std::pow(0.5, 2.5);
};

// Main function.
int
main(int argc, char * argv[])
{
  // Parse the input: `function` and `degree`.
  if (argc != 3) {
	std::cerr << "Usage: " << argv[0] << " function degree" << std::endl;
	return 1;
  }
  
  function_t function_type;
  if (std::string(argv[1]) == "sine")
	function_type = sine;
  else if (std::string(argv[1]) == "step")
	function_type = step;
  else {
	std::cerr << "Error: function must be 'sine' or 'step'" << std::endl;
	return 1;
  }

  unsigned int r;
  try {
	r = std::stoul(argv[2]);
  } catch (const std::invalid_argument &e) {
	  std::cerr << "Error: degree must be an integer" << std::endl;
	  return 1;
  }

  // This object is an utility that takes the errors (it can handle more than a
  // single error type) computed for different mesh sizes (h in the theory, in
  // this case we are going to test these inside `N_el_values`) and computes the
  // convergence rates. With this in place, we are going to see the convergence
  // orders in a quantitative way.
  ConvergenceTable table;

  const std::vector<unsigned int> N_el_values = {10, 20, 40, 80, 160, 320};
  const unsigned int              degree      = r;
  const auto mu = [](const Point<dim> & /*p*/) { return 1.0; };
  const auto f  = [function_type](const Point<dim> &p) {
	switch (function_type) {
	  case sine:	// Points 3 and 4.
		// RHS for u = sin(2pi x) => f = -u'' = 4pi^2 sin(2pi x)
		return 4.0 * M_PI * M_PI * std::sin(2.0 * M_PI * p[0]);
	  case step: // Point 5.
		if (p[0] < 0.5)
		  return 0.0;
		else
		  return -std::sqrt(p[0] - 0.5);
	}
  };

  const ExactSolution exact_solution(function_type);

  // The following file will collect the convergence data in CSV format for the
  // different mesh sizes (`N_el_values`). This file can used to plot useful
  // convergence graphs (see `/scripts/plot_convergence.py`). The plotting can
  // help to visualize the convergence behavior and have a qualitative point of
  // view on the errors.
  std::ofstream convergence_file("convergence.csv");
  convergence_file << "h,eL2,eH1" << std::endl;

  for (const unsigned int &N_el : N_el_values) {
	Poisson1D problem(N_el, degree, mu, f);

	problem.setup();
	problem.assemble();
	problem.solve();
	problem.output();

	const double h = 1.0 / N_el;

	const double error_L2 =
	problem.compute_error(VectorTools::L2_norm, exact_solution);
	const double error_H1 =
	problem.compute_error(VectorTools::H1_norm, exact_solution);

	// Add error information to the convergence table.
	table.add_value("h", h);
	table.add_value("L2", error_L2);
	table.add_value("H1", error_H1);

	// Add error value to the CSV file.
	convergence_file << h << "," << error_L2 << "," << error_H1 << std::endl;
  }

  // After collecting all the errors the convergence table can computes the
  // convergence rates. 
  table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
  // Formatting the columns of the table.
  table.set_scientific("L2", true);
  table.set_scientific("H1", true);

  table.write_text(std::cout);

  return 0;
}
