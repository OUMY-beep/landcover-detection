import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Multiclass Dice Loss.
    """
    def __init__(self, ignore_index=255, smooth=1e-5):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W)
        targets: (B, H, W)
        """
        num_classes = logits.size(1)
        
        # Apply softmax to logits
        probs = F.softmax(logits, dim=1)
        
        # Create one-hot encoding for targets, handling ignore_index
        valid_mask = targets != self.ignore_index
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0
        
        targets_one_hot = F.one_hot(targets_safe, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Zero out the ignored pixels in both probs and targets_one_hot
        valid_mask = valid_mask.unsqueeze(1).expand_as(probs)
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask
        
        dims = (0, 2, 3) # compute over batch, height, width
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        return 1.0 - dice_score.mean()


class CEDiceLoss(nn.Module):
    """
    Combination of Cross Entropy Loss and Dice Loss.
    """
    def __init__(self, ignore_index=255, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        loss_ce = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice
