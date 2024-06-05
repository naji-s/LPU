from matplotlib import pyplot as plt
import numpy as np

def plot_scores(scores_dict, best_epoch=0, loss_type='overall_loss'):


    # Plot data for each metric
    metrics = ['y_auc', 'y_accuracy', 'y_APS', loss_type]
    titles = ['AUC', 'Accuracy', 'y_APS', f'{loss_type}']

    num_plots = len(metrics)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, num_plots * 3))

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]

        for split in ['train', 'val']:
            epochs = scores_dict[split]['epochs']
            scores = scores_dict[split].get(metric, [None] * len(epochs))
            ax.plot(epochs, scores, label=f'{split} {metric}')

        # Mark minimum loss epoch on all subplots
        ax.axvline(best_epoch, color='r', linestyle='--', label='Min Total Loss')

        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.legend()

    plt.tight_layout()  # Adjust subplots to fit into figure area.
    plt.show()