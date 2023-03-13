python3 setup.py bdist_wheel; pip3 install --force-reinstall --no-deps dist/*.whl

# Clear
rm -r build
rm -r dist
rm -r *.egg-info
