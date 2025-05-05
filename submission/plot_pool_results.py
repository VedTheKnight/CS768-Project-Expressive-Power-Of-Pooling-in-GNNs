import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load results ===
df = pd.read_csv("experiment_results_partial_sweep.csv")

sns.set(style="whitegrid", font_scale=1.1)
palette = sns.color_palette("tab10")

def plot_param_variation(param, title, xlabel, filename):
    plt.figure(figsize=(8, 5))

    # Filter rows where other params are at default values
    if param == "pool_ratio":
        data = df[(df["num_layers_pre"] == 2) & (df["num_layers_post"] == 1)]
    elif param == "num_layers_pre":
        data = df[(df["pool_ratio"] == 0.1) & (df["num_layers_post"] == 1)]
    elif param == "num_layers_post":
        data = df[(df["pool_ratio"] == 0.1) & (df["num_layers_pre"] == 2)]
    else:
        raise ValueError("Invalid param")

    sns.lineplot(
        data=data,
        x=param,
        y="test_acc",
        hue="pooling",
        marker="o",
        palette=palette,
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Test Accuracy")
    plt.legend(title="Pooling")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.show()


# === Generate plots ===
plot_param_variation("pool_ratio", "Effect of Pool Ratio", "Pool Ratio", "pool_ratio_vs_acc.png")
plot_param_variation("num_layers_pre", "Effect of Num Pre-Layers", "Pre-GNN Layers", "pre_layers_vs_acc.png")
plot_param_variation("num_layers_post", "Effect of Num Post-Layers", "Post-GNN Layers", "post_layers_vs_acc.png")
