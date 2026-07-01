
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import segmentation_models_pytorch as smp
from PIL import Image
# imports 2
import os
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys

sys.stdout = open("training_log.txt", "w")

# Fix seed for reproducibility
SEED_VALUE = 42
torch.manual_seed(SEED_VALUE)

# Image config
IMG_HEIGHT = 256
IMG_WIDTH  = 256
BATCH_SIZE = 4


CLASS_NAMES = [
    "background",    
    "hat",           
    "hair",          
    "glove",         
    "sunglasses",    
    "upper_clothes", 
    "dress",         
    "coat",          
    "socks",         
    "pants",         
    "jumpsuits",     
    "scarf",         
    "skirt",         
    "face",          
    "left_arm",      
    "right_arm",     
    "left_leg",      
    "right_leg",     
    "left_shoe",     
    "right_shoe",    
]
NUM_CLASSES = len(CLASS_NAMES)



class FashionSegDataset(Dataset):
    """Semantic segmentation dataset for fashion / human parsing.

    Expects the directory layout:
        root_dir/
            images/   ← RGB .jpg photos
            masks/    ← greyscale .png masks  (pixel value = class index)

    Outputs a (C,H,W) image tensor
    and a (H,W) long mask tensor instead of a multi-hot label vector (initial thought).
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir   = root_dir
        self.transform  = transform
        self.image_dir  = os.path.join(root_dir, "images")
        self.mask_dir   = os.path.join(root_dir, "masks")
        self.classes    = CLASS_NAMES

        # Collect matched image/mask pairs
        self.image_paths = []
        self.mask_paths  = []

        for fname in sorted(os.listdir(self.image_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path  = os.path.join(self.image_dir, fname)
                mask_name = os.path.splitext(fname)[0] + ".png"
                mask_path = os.path.join(self.mask_dir, mask_name)
                if os.path.exists(mask_path):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask  = np.array(Image.open(self.mask_paths[idx]))   

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]            # (3,H,W) float tensor
            mask  = augmented["mask"].long()      # (H,W)   long tensor
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask  = torch.from_numpy(mask).long()

        return image, mask


# ── TRANSFORMS ───────────────────────────────────────────────────────────────
# train_transform adds augmentation; val_transform only resizes + normalises.

def get_train_transform():
    return A.Compose([
        A.Resize(IMG_HEIGHT, IMG_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15, p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transform():
    return A.Compose([
        A.Resize(IMG_HEIGHT, IMG_WIDTH),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])



if __name__ == "__main__":
    train_dataset = FashionSegDataset(
    root_dir="dataset/Train",
    transform=get_train_transform()
        )
    valid_dataset = FashionSegDataset(
    root_dir="dataset/Valid",
    transform=get_val_transform()
    )
    test_dataset  = FashionSegDataset(
    root_dir="dataset/Test",
    transform=get_val_transform()
    )


if __name__ == "__main__":
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


##### MODEL architecture
# U-Net with ResNet34 encoder 
# ResNet50 becomes the encoder
# a pixel-level class map 

DROPOUT      = 0.3   
DENSE_SIZE   = 256  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name    = "resnet34",      
    encoder_weights = "imagenet",    
    in_channels     = 3,
    classes         = NUM_CLASSES,    
    decoder_use_batchnorm = True,
)
model = model.to(device)

if __name__ == "__main__":
    print("Dataset Information:")
    print(f"  Train samples : {len(train_dataset)}")
    print(f"  Valid samples : {len(valid_dataset)}")
    print(f"  Test  samples : {len(test_dataset)}")
print(f"  Number of classes : {NUM_CLASSES}")
print(f"  Batch size        : {BATCH_SIZE}")

print("\nClass index mapping:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {i:>2}: {name}")


##### TRAINING
#input: model, train_loader, valid_loader, num_epochs, device, save_path
def train_model_segmentation(model, train_loader, valid_loader,
                              num_epochs, device, save_path):
    """Train U-Net segmentation model with early stopping on validation loss.


      - tracks best validation loss
      - saves best weights to disk
      - returns model loaded with best weights

    Uses CrossEntropyLoss (one class per pixel) 
    """

    criterion = nn.CrossEntropyLoss(ignore_index=255)   # 255 = unlabelled pixel
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_loss   = float("inf")
    best_model_state = None
    patience_counter = 0
    EARLY_STOP_PATIENCE = 15

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
    
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)           
            loss    = criterion(outputs, masks)  

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in valid_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()

        avg_val_loss = val_loss / len(valid_loader)
        scheduler.step(avg_val_loss)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1:>3}/{num_epochs}]  "
              f"Train Loss: {avg_train_loss:.4f}  |  "
              f"Val Loss: {avg_val_loss:.4f}")

        
        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_model_state, save_path)
            print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}.")
                break
       


    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model weights for evaluation.")

   
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("CrossEntropy Loss")
    plt.title("Training Curve — Fashion Segmentation")
    plt.legend(); plt.tight_layout()
    plt.savefig("training_curve.png")
    plt.close()

    return model 
if __name__ == "__main__":
    print("Dataset Information:")
    print(f"  Train samples : {len(train_dataset)}")
    print(f"  Valid samples : {len(valid_dataset)}")
    print(f"  Test  samples : {len(test_dataset)}")
    print(f"  Number of classes : {NUM_CLASSES}")
    print(f"  Batch size        : {BATCH_SIZE}")

    print("\nClass index mapping:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i:>2}: {name}")
if __name__ == "__main__":
    NUM_EPOCHS      = 20
    model_save_path = "fashion_seg_best.pth"
    model = train_model_segmentation(
        model, train_loader, valid_loader, NUM_EPOCHS, device, model_save_path
    )

sys.stdout.close()
sys.stdout = sys.__stdout__
