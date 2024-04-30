def plot_scores(scores_dict, loss_type='L_mpe'):
    fig, ax = plt.subplots(5, 1, figsize=(10, 10))
    # Calculate the index of the minimum Total Loss
    min_loss_index = np.argmin(scores_dict['val'][loss_type])  # Index of minimum Total Loss

    # AUC Plot
    ax[0].plot(scores_dict['val']['y_auc'], label='val AUC')
    ax[0].plot(min_loss_index, scores_dict['val']['y_auc'][min_loss_index], 'rx', markersize=10, label='Min Total Loss')  # Mark the min Total Loss point
    ax[0].set_title('AUC')
    ax[0].legend()

    # Accuracy Plot
    ax[1].plot(scores_dict['val']['y_accuracy'], label='val Accuracy')
    ax[1].plot(min_loss_index, scores_dict['val']['y_accuracy'][min_loss_index], 'rx', markersize=10, label='Min Total Loss')  # Mark the min Total Loss point
    ax[1].set_title('val Accuracy')

    # Test AUC and Accuracy Plot
    ax[2].plot(scores_dict['test']['y_auc'], label='Test AUC')
    ax[2].plot(min_loss_index, scores_dict['test']['y_auc'][min_loss_index], 'rx', markersize=10, label='Min Total Loss')  # Mark the min point
    ax[2].set_title('Test AUC')

    ax[3].plot(scores_dict['test']['y_accuracy'], label='Test Accuracy')
    ax[3].plot(min_loss_index, scores_dict['test']['y_accuracy'][min_loss_index], 'rx', markersize=10, label='Min Total Loss')  # Mark the min point
    ax[3].set_title(f'Test Accuracy')

    ax[4].plot(scores_dict['train'][loss_type], label=f'train {loss_type}')
    ax[4].plot(min_loss_index, scores_dict['train'][loss_type][min_loss_index], 'rx', markersize=10, label='Min Total Loss')  # Mark the min point
    ax[4].set_title(f'train {loss_type}')

    ax[4].plot(scores_dict['val'][loss_type], label=f'val {loss_type}')
    ax[4].plot(min_loss_index, scores_dict['val'][loss_type][min_loss_index], 'rx', markersize=10, label='Min Total Loss')  # Mark the min point
    ax[4].set_title(f'val {loss_type}')

    ax[4].plot(scores_dict['test'][loss_type], label=f'test {loss_type}')
    ax[4].plot(min_loss_index, scores_dict['test'][loss_type][min_loss_index], 'rx', markersize=10, label='Min Total Loss')  # Mark the min point
    ax[4].set_title(f'Test {loss_type}')


    plt.tight_layout()  # Adjust subplots to fit into figure area.
    plt.legend()
    plt.show()