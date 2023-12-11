import matplotlib.pyplot as plt
import pandas as pd
import time


def make_finetune_curves(primary_hist, secondary_hist) -> None:
    """
    Make the curves for finetuned models.

    :param primary_hist: The history object of the untuned model.
    :param secondary_hist: The history object of the tuned model.
    """

    history_subspecies_df = pd.DataFrame(secondary_hist)
    break_point = history_subspecies_df.index.values.max()
    history_subspecies2_df = pd.DataFrame(primary_hist)
    history_subspecies2_df.set_index(pd.Index(range(break_point, break_point + len(history_subspecies2_df))),
                                     inplace=True)
    last_index = history_subspecies2_df.index.values.max()

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(16, 5))

    # Plot for ax1
    ax1.plot(history_subspecies_df["loss"], color='blue', label='Training Loss')
    ax1.plot(history_subspecies2_df["loss"], color='blue')

    ax1.plot(history_subspecies_df["val_loss"], color='red', label='Validation Loss')
    ax1.plot(history_subspecies2_df["val_loss"], color='red')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.axvline(x=break_point, color='grey', linestyle='dashdot')

    ax1.axvspan(0, break_point, facecolor='lightblue', alpha=0.2, label="Freezed Epochs")
    ax1.axvspan(break_point, last_index, facecolor='red', alpha=0.1, label="Unfreezed Epochs")

    ax1.set_xlim(left=0, right=last_index)
    ax1.legend()

    # Plot for ax2
    ax2.plot(history_subspecies_df["sparse_categorical_accuracy"], color='blue', label='Training Accuracy')
    ax2.plot(history_subspecies2_df["sparse_categorical_accuracy"], color='blue', )

    ax2.plot(history_subspecies2_df["val_sparse_categorical_accuracy"], color='red', label='Validation Accuracy')
    ax2.plot(history_subspecies_df["val_sparse_categorical_accuracy"], color='red')

    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')

    ax2.axvline(x=break_point, color='grey', linestyle='dashdot')

    ax2.axvspan(0, break_point, facecolor='lightblue', alpha=0.2, label="Freezed Epochs")
    ax2.axvspan(break_point, last_index, facecolor='red', alpha=0.1, label="Unfreezed Epochs")
    ax2.legend()

    ax2.annotate('Un-Freezed Network', xy=(10, max(ax1.get_ylim())), xytext=(15, max(ax1.get_ylim())),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)
    ax2.set_xlim(left=0, right=last_index)

    background_color = '#F8F1E2'  # Replace this with your hexadecimal color value
    fig.patch.set_facecolor(background_color)

    name = time.time()
    plt.savefig(f"../data/finetune_curves{name}.png", dpi=300)
