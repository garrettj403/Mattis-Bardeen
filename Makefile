# Makefile for Mattis-Bardeen module
#
# Makefile examples that I drew from:
# https://github.com/pjz/mhi/blob/master/Makefile
# http://krzysztofzuraw.com/blog/2016/makefiles-in-python-projects.html
#

CODE_PATH=./mattisbardeen.py
TEST_PATH=./test_mattisbardeen.py

all: test clean

# Run tests (with Py.Test) ---------------------------------------------------

test:
	py.test $(TEST_PATH) --verbose --doctest-modules --color=yes

cov:
	py.test $(TEST_PATH) --verbose --doctest-modules --color=yes --cov=$(CODE_PATH) --cov-report=term-missing

cov-report: 
	py.test $(TEST_PATH) --verbose --doctest-modules --color=yes --cov=$(CODE_PATH) --cov-report=html
	open htmlcov/index.html

clean-test:
	find . -name 'htmlcov' -exec rm -rf {} +
	
# Build ----------------------------------------------------------------------

build:
	python setup.py sdist

upload:
	twine upload dist/*

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf QMix.egg-info/
	rm MANIFEST

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

.PHONY: all test clean-test clean clean-bytecode clean-hidden
