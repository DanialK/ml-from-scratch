## Setup Environment
- `python3.9 -m 'venv' venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`

# To use Jupyter
- `ipython kernel install --name "local-venv" --user`


# To run the tests
- In PyCharm, ensure to include `./src` as a source root and run the tests
- In VSCode or other IDES ensure to add `./src` to `PYTHONPATH`
  - `PYTHONPATH="${PYTHONPATH}:$(pwd)/src"`
- Execute tests `PYTHONPATH="${PYTHONPATH}:$(pwd)/src" python -m pytest`