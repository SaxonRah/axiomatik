1. Change version number in `setup.py` and `__init__.py`
2. Delete `axiomatik.egg-info` and `dist`
3. Run `python -m build`
4. Run `python -m twine upload dist/* --verbose`