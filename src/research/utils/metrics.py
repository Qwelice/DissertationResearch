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