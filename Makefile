SOURCE_DIRS = src
TEST_DIRS = tests
SOURCE_AND_TEST_DIRS = $(SOURCE_DIRS) $(TEST_DIRS)
PREFIX = poetry run

.PHONY: format lint-fix fix format-check lint pyright test

all: format-check lint pyright test

format:
	$(PREFIX) ruff -e --fix-only --select I001 $(SOURCE_AND_TEST_DIRS)
	$(PREFIX) black $(SOURCE_AND_TEST_DIRS)

lint-fix:
	$(PREFIX) ruff -e --fix-only $(SOURCE_AND_TEST_DIRS)

fix: lint-fix
	$(PREFIX) black $(SOURCE_AND_TEST_DIRS)

format-check:
	@($(PREFIX) ruff --select I001 $(SOURCE_AND_TEST_DIRS)) && ($(PREFIX) black --check $(SOURCE_AND_TEST_DIRS)) || (echo "run \"make format\" to format the code"; exit 1)

lint:
	@($(PREFIX) ruff $(SOURCE_AND_TEST_DIRS)) || (echo "run \"make lint-fix\" to fix some lint errors automatically"; exit 1)

pyright:
	$(PREFIX) pyright

test:
	$(PREFIX) python -m pytest $(TEST_DIRS)
