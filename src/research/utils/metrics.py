import torch


def discriminator_accuracy(discriminator_preds, is_real: bool, threshold: float=0.5):
    """
    Args:
        discriminator_preds: List[List[Tensor]]
        is_real: bool which logits is using: fake or real
        threshold: accuracy threshold
    Returns:
        accuracy: float scalar tensor
    """

    total = 0
    correct = 0

    for logits in discriminator_preds:
        for logit in logits:
            preds = torch.sigmoid(logit) > threshold
            targets = (torch.ones_like(preds) if is_real else torch.zeros_like(preds)).to(preds.device)
            correct += (preds == targets).sum().item()
            total += preds.numel()

    return correct / total if total > 0 else 0.0


def compute_iou_voxels(pred_voxels, target_voxels, threshold=0.5):
    """
    Вычисляет IoU между двумя списками воксельных тензоров.

    Аргументы:
        pred_voxels (List[Tensor]): список предсказанных вокселей, каждый тензор размера [D, H, W] или [B, D, H, W]
        target_voxels (List[Tensor]): список таргетных вокселей того же размера
        threshold (float): порог бинаризации (по умолчанию 0.5)

    Возвращает:
        Tensor: средний IoU по всем парам
    """
    assert len(pred_voxels) == len(target_voxels), "Списки должны быть одинаковой длины"
    iou_total = 0.0
    for pred, target in zip(pred_voxels, target_voxels):
        pred_bin = (pred > threshold).float()
        target_bin = (target > threshold).float()

        intersection = (pred_bin * target_bin).sum()
        union = ((pred_bin + target_bin) >= 1).float().sum()
        iou = intersection / (union + 1e-6)
        iou_total += iou
    return iou_total / len(pred_voxels)