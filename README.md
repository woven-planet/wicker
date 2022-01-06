# Wicker

Wicker is an open source framework for Machine Learning dataset storage and serving developed at Woven Planet L5.

# Usage

Refer to the [Wicker documentation's Getting Started page](https://woven-planet.github.io/wicker/getstarted.html) for more information.

# Development

To develop on Wicker to contribute, set up your local environment as follows:

1. Create a new virtual environment
2. Do a `pip install -r dev-requirements.txt` to install the development dependencies
3. Run `make test` to run all unit tests
4. Run `make lint-fix` to fix all lints and `make lint` to check for any lints that must be fixed manually
5. Run `make type-check` to check for type errors

To contribute a new plugin to have Wicker be compatible with other technologies (e.g. Kubernetes, Ray, AWS batch etc):

1. Add your plugin into the `wicker.plugins` module as an appropriately named module
2. If your new plugin requires new external dependencies:
    1. Add a new extra-requires entry to `setup.cfg`
    2. Update `dev-requirements.txt` with any necessary dependencies to run your module in unit tests
3. Write a unit test in `tests/` to test your module
