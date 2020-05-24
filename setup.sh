cd ./model/networks/block_extractor
python setup.py clean --all install --user

cd ..
cd local_attn_reshape
python setup.py clean --all install --user

cd ..
cd resample2d_package
python setup.py clean --all install --user
