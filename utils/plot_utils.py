from matplotlib import pyplot as plt
import numpy as np

def plot_scores(scores_dict, loss_type=None):
    num_plots = 3
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 8))

    # Calculate the epoch of the minimum Total Loss
    val_epochs = scores_dict['val']['epochs']
    min_loss_epoch = val_epochs[np.argmin(scores_dict['val'][loss_type])]
    min_loss_score = scores_dict['val'][loss_type][val_epochs.index(min_loss_epoch)]

    # Plot data for each metric
    metrics = ['y_auc', 'y_accuracy', loss_type]
    titles = ['AUC', 'Accuracy', f'{loss_type}']

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]

        for split in ['train', 'val', 'test']:
            epochs = scores_dict[split]['epochs']
            scores = scores_dict[split].get(metric, [None] * len(epochs))
            ax.plot(epochs, scores, label=f'{split} {metric}')

        # Mark minimum loss epoch on all subplots
        ax.axvline(min_loss_epoch, color='r', linestyle='--', label='Min Total Loss')

        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.legend()

    plt.tight_layout()  # Adjust subplots to fit into figure area.
    plt.show()