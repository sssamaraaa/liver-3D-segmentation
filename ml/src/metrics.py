import torch
import torch.nn as nn


def dice_coef(pred, target, eps=1e-6, ignore_empty=True):
    """Dice, excluding completely empty targets (gt.sum()==0)"""
    pred = pred.contiguous().view(pred.shape[0], -1) 
    target = target.contiguous().view(target.shape[0], -1)

    if ignore_empty:
        valid_mask = target.sum(dim=1) > 0 
        if valid_mask.any():
            pred = pred[valid_mask] 
            target = target[valid_mask]
        else:
            return torch.tensor(1.0, device=pred.device) 

    intersection = (pred * target).sum(-1) 
    denomination = pred.sum(-1) + target.sum(-1) 
    dice = (2 * intersection) / (denomination + eps) 
    return dice.mean()

class DiceBCELoss(nn.Module):
    def __init__(self, weight_bce=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.weight_bce = weight_bce 

    def forward(self, logits, targets):
        bce = self.bce(logits, targets) 
        probs = torch.sigmoid(logits) 
        dice = dice_coef(probs, targets)
        return self.weight_bce * bce + (1 - self.weight_bce) * (1 - dice) 
    
def dice(inter, p_sum, g_sum, eps=1e-6):
    denom = p_sum + g_sum + eps
    return 2.0 * inter / denom

def iou(inter, p_sum, g_sum, eps=1e-6):
    denom = p_sum + g_sum - inter + eps
    return inter / denom

def precision(inter, p_sum, eps=1e-6):
    return inter / (p_sum + eps)

def recall(inter, g_sum, eps=1e-6):
    return inter / (g_sum + eps)

def f1_score(prec, rec, eps=1e-6):
    return 2 * prec * rec / (prec + rec + eps)