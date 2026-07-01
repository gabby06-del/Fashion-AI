import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from fashion_eval_metrics import evaluate_model_segmentation
from fashion_combined import (FashionSegDataset, get_val_transform,  NUM_CLASSES, CLASS_NAMES)
if __name__ == "__main__":
    checkpoint_path = "fashion_seg_best.pth"
    test_root       = "dataset/Test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = FashionSegDataset(test_root, transform=get_val_transform())
    test_loader  = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)


    model = smp.Unet(
    encoder_name    = "resnet50",
    encoder_weights = None,        
    in_channels     = 3,
    classes         = NUM_CLASSES,
    decoder_use_batchnorm = True,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)


    metrics = evaluate_model_segmentation(
    model        = model,
    test_loader  = test_loader,
    device       = device,
    dataset_name = "Test",
    num_classes  = NUM_CLASSES,
)   