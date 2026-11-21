# Makefile for Mattis-Bardeen module
#
# Makefile examples that I drew from:
# https://github.com/pjz/mhi/blob/master/Makefile
# http://krzysztofzuraw.com/blog/2016/makefiles-in-python-projects.html
#

all: test clean

# Run tests (with Py.Test) ---------------------------------------------------

# Test paths and --doctest-modules are configured in pyproject.toml
test:
	pytest --verbose --color=yes

cov:
	pytest --verbose --color=yes --cov=mattisbardeen --cov-report=term-missing

cov-report:
	pytest --verbose --color=yes --cov=mattisbardeen --cov-report=html
	open htmlcov/index.html

clean-test:
	find . -name 'htmlcov' -exec rm -rf {} +
	
# Build ----------------------------------------------------------------------

build: clean-build
	python -m build
	twine check dist/*

upload-test: build
	twine upload --repository testpypi dist/*

upload: build
	twine upload dist/*

clean-build:
	rm -rf build/ dist/ *.egg-info/

# Clean ----------------------------------------------------------------------

clean-bytecode:
	find . -name '*.pyc' -exec rm -f {} +

clean-hidden:
	find . -name '.coverage' -exec rm -rf {} +
	find . -name '.cache' -exec rm -rf {} +
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '__pycache__\ \(1\)' -exec rm -rf {} +

clean: clean-bytecode clean-hidden clean-test clean-build

# Misc -----------------------------------------------------------------------

.PHONY: all test cov cov-report build upload-test upload clean-build clean-test clean clean-bytecode clean-hidden
