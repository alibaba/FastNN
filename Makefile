PYTHON ?= python

.PHONY: lint
lint:
	pip install pylint==2.12.2
	@$(PYTHON) -m pylint \
		--rcfile=.pylintrc --output-format=parseable --jobs=8 \
		$(shell find ./ -type f -name '*.py')
