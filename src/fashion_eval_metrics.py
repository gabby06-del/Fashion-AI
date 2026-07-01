import torch
import numpy as np

def evaluate_model_segmentation(model, test_loader, device,dataset_name="Test",num_classes=20, ignore_index=255):

    model.eval()
    criterion   = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    total_loss  = 0.0
    total_px, correct_px = 0, 0
    tp = np.zeros(num_classes, dtype=np.float64)
    fp = np.zeros(num_classes, dtype=np.float64)
    fn = np.zeros(num_classes, dtype=np.float64)
    zero_iou_count, false_pos_count, total_cls_inst = 0, 0, 0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, masks).item()
            preds = outputs.argmax(dim=1)

            valid = masks != ignore_index
            correct_px += (preds[valid] == masks[valid]).sum().item()
            total_px   += valid.sum().item()

            p = preds.cpu().numpy().reshape(-1)
            m = masks.cpu().numpy().reshape(-1)
            v = (m != ignore_index)
            p, m = p[v], m[v]

            for cls in range(num_classes):
                pc = (p == cls); mc = (m == cls)
                t  = np.logical_and(pc, mc).sum()
                f  = np.logical_and(pc, ~mc).sum()
                n  = np.logical_and(~pc, mc).sum()
                tp[cls] += t; fp[cls] += f; fn[cls] += n
                if mc.sum() > 0:
                    total_cls_inst += 1
                    if t == 0: zero_iou_count += 1
                if pc.sum() > 0 and mc.sum() == 0:
                    false_pos_count += 1
    avg_loss  = total_loss / len(test_loader)
    pixel_acc = 100.0 * correct_px / max(total_px, 1)
    iou  = tp / np.maximum(tp + fp + fn, 1e-6)
    dice = (2*tp) / np.maximum(2*tp + fp + fn, 1e-6)
    present = (tp + fn) > 0
    mean_iou  = iou[present].mean()  if present.any() else 0.0
    mean_dice = dice[present].mean() if present.any() else 0.0
    zero_iou_rate  = 100.0 * zero_iou_count  / max(total_cls_inst, 1)
    false_pos_rate = 100.0 * false_pos_count / max(total_cls_inst, 1)

    from fashion_combined import CLASS_NAMES
    print(f"\n{dataset_name} - Segmentation Evaluation:")
    print("="*60)
    print(f"\nCORE METRICS:")
    print(f"  Loss           : {avg_loss:.4f}")
    print(f"  Pixel Accuracy : {pixel_acc:.2f}%")
    print(f"  Mean IoU       : {mean_iou:.4f}")
    print(f"  Mean Dice      : {mean_dice:.4f}")
    print(f"\nERROR ANALYSIS:")
    print(f"  Zero-IoU Rate       : {zero_iou_rate:.2f}%  (class present, never predicted)")
    print(f"  False-Positive Rate : {false_pos_rate:.2f}%  (class predicted, absent in GT)")
    print(f"\nPER-CLASS IoU & DICE:")
    for i, name in enumerate(CLASS_NAMES):
        if present[i]:
            print(f"  {name:<20} IoU: {iou[i]:.4f}  Dice: {dice[i]:.4f}")
        else:
            print(f"  {name:<20} (not in test set)")
    print("="*60)

    return {"loss":avg_loss,"pixel_accuracy":pixel_acc,"mean_iou":mean_iou,
            "mean_dice":mean_dice,"iou_per_class":dict(zip(CLASS_NAMES,iou.tolist())),
            "dice_per_class":dict(zip(CLASS_NAMES,dice.tolist())),
            "zero_iou_rate":zero_iou_rate,"false_pos_rate":false_pos_rate}
   