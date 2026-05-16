This folder contains the LocPick documentation sources for the `UDST/choicemodels` repository.

## How it works

Sphinx generates the HTML files. GitHub Pages hosts them from the `gh-pages` branch. Maintainers update the rendered documentation manually.

## Editing the documentation

The files in `docs/source`, along with docstrings in the source code, determine what appears in the rendered documentation. Here's a [good tutorial](https://pythonhosted.org/an_example_pypi_project/sphinx.html) for Sphinx.

## Previewing changes locally

Install the copy of LocPick that the documentation should reflect, then install the documentation tools.

```bash
pip install .
pip install sphinx sphinx_rtd_theme
```

Build the documentation. There should be status messages and warnings, but no errors.

```bash
cd docs
sphinx-build -b html source build
```

The HTML files will show up in `docs/build/`.

## Uploading changes

Clone a second copy of the repository and check out the `gh-pages` branch. Copy over the updated HTML files, commit them, and push the changes to GitHub.
