conda create lwg-proj python=3.10 -y
conda activate lwg-proj

conda install pytorch==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
