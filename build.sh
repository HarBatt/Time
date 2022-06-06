python3 -V
pip3 install wheel --upgrade
python3 setup.py bdist_wheel
mv -fv dist/*.whl .
rm -r build
rm -r tsgeneration.egg-info
rm -r .eggs
rm -r dist
rm -f .DS_Store
