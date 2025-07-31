import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_loss(df, title="Loss per Epoch", save_path=None):
    """
    Строит график потерь по эпохам (train/val).

    :param df: pandas.DataFrame со столбцами 'epoch', 'train_loss', 'val_loss'
    :param title: заголовок графика
    :param save_path: путь для сохранения графика (например, 'loss_plot.png')
    """
    sns.set(style="whitegrid", font_scale=1.2)

    plt.figure(figsize=(8, 4))

    sns.lineplot(x='epoch', y='train_loss', data=df, marker='o', label='Train Loss')
    sns.lineplot(x='epoch', y='val_loss', data=df, marker='s', label='Validation Loss')

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"График сохранён в {save_path}")
    else:
        plt.show()
