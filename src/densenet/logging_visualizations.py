import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def visualize_learning_logs(path_train_learning_logs, path_test_learning_logs, path_log_dir):
    # Read the train and test learning logs from CSV files
    df_train_logs = pd.read_csv(path_train_learning_logs)
    df_test_logs = pd.read_csv(path_test_learning_logs)

    # Compute the average loss and standard deviation for each epoch
    train_epoch_loss = df_train_logs.groupby('epoch')['loss'].agg(['mean', 'std']).reset_index()
    test_epoch_loss = df_test_logs.groupby('epoch')['loss'].agg(['mean', 'std']).reset_index()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the average train loss with error bars
    ax.errorbar(train_epoch_loss['epoch'], train_epoch_loss['mean'], yerr=train_epoch_loss['std'],
                fmt='o-', capsize=4, label='Train Loss', color='blue', zorder=1, alpha=0.7)

    # Plot the average test loss with error bars
    ax.errorbar(test_epoch_loss['epoch'], test_epoch_loss['mean'], yerr=test_epoch_loss['std'],
                fmt='o-', capsize=4, label='Test Loss', color='orange', zorder=2, alpha=0.7)

    # Set the plot title and labels
    ax.set_title('Average Loss per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    xticks = np.arange(0, len(train_epoch_loss['epoch']), 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(train_epoch_loss['epoch'])

    # Add a legend
    ax.legend()

    # Save the plot to a file
    path_figure = os.path.join(path_log_dir, 'learning_logs_visualization.png')
    fig.savefig(path_figure, dpi=300, bbox_inches='tight')

    # Close the figure to free up memory
    plt.close(fig)

    print(f'Learning logs visualization saved to {path_figure}')