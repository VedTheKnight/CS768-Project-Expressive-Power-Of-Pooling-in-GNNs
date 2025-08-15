# The Expressive Power of Pooling in GNNs

## Environment Setup

```bash
# Create and activate conda environment
conda create lwg-proj python=3.10 -y
conda activate lwg-proj

# Install PyTorch and CUDA
conda install pytorch==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PyTorch Geometric and dependencies
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## How to Run

1. Navigate to the `submission/` directory

2. To run all experiments:
   ```bash
   python run_pool_experiments.py
   python plot_pool_results.py
   ```

3. To run individual pooling methods:
   ```bash
   python main.py --pooling <pooling-method>
   ```
   
   Additional options available:
   - Number of epochs
   - Number of pre and post layers
   - And more... 

## Structure 

1. The directory `submissions/scripts/pooling` contains all the pooling methods we have implemented
2. `submission/nn_model.py` is the file that defines the entire model architecture, incorporating the GIN and Pooling layers
3. `submission/run_pool_experiments.py` and `submission/plot_pool_results.py` contains the code to run the experiments for the results displayed in section 5 of the report
4. `submission/main.py` is the file to test out each individual pooling method by itself
5. `submission/data` contains the EXPWL1 dataset
6. `submission/results` contains the results of our experiments, individual pngs of the graphs as well as the numerical results

## References

We referred to the Github implementation of the paper : https://github.com/FilippoMB/The-expressive-power-of-pooling-in-GNNs. For the dataset as well as the train-test pipeline.
