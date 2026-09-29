
I want to convert this repository into a Python package that can be published to PyPI and installed using `pip install tokengov`.

Please do the following step-by-step:

1. **Check or Create `pyproject.toml`**: Check if there is a `pyproject.toml` or `setup.py` file. If not, create a modern `pyproject.toml` for this repository using `setuptools.build_meta`. Ensure the package name is set to `<your-unique-package-name>` (this must be globally unique on PyPI to avoid 403 Forbidden errors).
2. **Clean Old Builds**: Run a command to forcefully delete any existing `dist` directory in the repository to prevent uploading stale artifacts.
3. **Build the Package**: Run `python -m build` to generate the `.tar.gz` and `.whl` files.
4. **Publish to PyPI**: I am providing my Twine credentials as environment variables to avoid interactive prompt issues. Run the Twine upload command.

Here are my credentials for the upload:

- Username: `__token__`
- Password: `from the key value of "PYPI_TOKEN" in .env file`

Please execute these steps directly for me, and let me know if there are any naming conflicts or errors!
