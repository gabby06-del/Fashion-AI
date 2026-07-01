import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import segmentation_models_pytorch as smp
from fashion_combined import test_dataset, NUM_CLASSES, CLASS_NAMES

if __name__ == "__main__":
    checkpoint_path = "fashion_seg_best.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smp.Unet(encoder_name="resnet50", encoder_weights=None,
                     in_channels=3, classes=NUM_CLASSES,
                     decoder_use_batchnorm=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    COLORS = plt.cm.tab20(np.linspace(0, 1, NUM_CLASSES))

    def visualize(idx=0, save=True):
        image, true_mask = test_dataset[idx]

        with torch.no_grad():
            output    = model(image.unsqueeze(0).to(device))
            pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()

        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img  = image.permute(1,2,0).numpy()
        img  = (img * std + mean).clip(0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        axes[0].imshow(img)
        axes[0].set_title("Input image")
        axes[1].imshow(true_mask.numpy(), cmap="tab20", vmin=0, vmax=NUM_CLASSES)
        axes[1].set_title("Ground truth")
        axes[2].imshow(pred_mask, cmap="tab20", vmin=0, vmax=NUM_CLASSES)
        axes[2].set_title("Prediction")
        for ax in axes:
            ax.axis("off")

        patches = [mpatches.Patch(color=COLORS[i], label=CLASS_NAMES[i])
                   for i in range(NUM_CLASSES)]
        fig.legend(handles=patches, loc="lower center",
                   ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.08))
        plt.tight_layout()

        if save:
            plt.savefig(f"prediction_{idx}.png", bbox_inches="tight", dpi=150)
            print(f"Saved prediction_{idx}.png")
        plt.show()
        plt.close()

    for i in range(5):
        visualize(i)