import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the CSV
df = pd.read_csv('experiment_results_partial_sweep.csv')  # Replace with your actual filename

# Optional: average over repeats
df_grouped = df.groupby(['pooling', 'pool_ratio', 'num_layers_pre', 'num_layers_post'], as_index=False).agg({
    'test_acc': 'mean',
    'time_sec': 'mean'
})

# --- Helper function to plot and save ---
import pandas as pd
import matplotlib.pyplot as plt

def plot_metric(df, x, y, title, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 6))
    df = df.copy()
    df[x] = pd.to_numeric(df[x], errors='coerce')
    df[y] = pd.to_numeric(df[y], errors='coerce')
    df = df.dropna(subset=[x, y, 'pooling'])

    for pooling_method, group in df.groupby('pooling'):
        sorted_group = group.sort_values(by=x)
        plt.plot(
            sorted_group[x].values, 
            sorted_group[y].values, 
            marker='o', 
            label=pooling_method
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(title='Pooling')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")



# 1. Vary pool_ratio (fixed layers)
fixed_pre, fixed_post = 2, 1
df_ratio = df_grouped[(df_grouped['num_layers_pre'] == fixed_pre) & (df_grouped['num_layers_post'] == fixed_post)]

plot_metric(df_ratio, 'pool_ratio', 'test_acc',
            f'Accuracy vs Pool Ratio (Pre={fixed_pre}, Post={fixed_post})',
            'Pool Ratio', 'Test Accuracy', 'accuracy_vs_pool_ratio.png')

plot_metric(df_ratio, 'pool_ratio', 'time_sec',
            f'Time vs Pool Ratio (Pre={fixed_pre}, Post={fixed_post})',
            'Pool Ratio', 'Time (sec)', 'time_vs_pool_ratio.png')


# 2. Vary num_layers_pre (fixed ratio & post)
fixed_ratio, fixed_post = 0.1, 1
df_pre = df_grouped[(df_grouped['pool_ratio'] == fixed_ratio) & (df_grouped['num_layers_post'] == fixed_post)]

plot_metric(df_pre, 'num_layers_pre', 'test_acc',
            f'Accuracy vs Num Pre-layers (Ratio={fixed_ratio}, Post={fixed_post})',
            '# Pre GIN Layers', 'Test Accuracy', 'accuracy_vs_num_pre_layers.png')

plot_metric(df_pre, 'num_layers_pre', 'time_sec',
            f'Time vs Num Pre-layers (Ratio={fixed_ratio}, Post={fixed_post})',
            '# Pre GIN Layers', 'Time (sec)', 'time_vs_num_pre_layers.png')


# 3. Vary num_layers_post (fixed ratio & pre)
fixed_ratio, fixed_pre = 0.1, 2
df_post = df_grouped[(df_grouped['pool_ratio'] == fixed_ratio) & (df_grouped['num_layers_pre'] == fixed_pre)]

plot_metric(df_post, 'num_layers_post', 'test_acc',
            f'Accuracy vs Num Post-layers (Ratio={fixed_ratio}, Pre={fixed_pre})',
            '# Post GIN Layers', 'Test Accuracy', 'accuracy_vs_num_post_layers.png')

plot_metric(df_post, 'num_layers_post', 'time_sec',
            f'Time vs Num Post-layers (Ratio={fixed_ratio}, Pre={fixed_pre})',
            '# Post GIN Layers', 'Time (sec)', 'time_vs_num_post_layers.png')
