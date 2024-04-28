import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class QWKLoss(nn.Module):
    def __init__(self, n_classes=3, eps=1e-7):
        super().__init__()
        
        self.n_classes = n_classes
        self.eps = eps
        
        W = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(n_classes):
                W[i,j] = (i-j)**2
        
        self.W = torch.from_numpy(W.astype(np.float32))
    
    def forward(self, preds, labels):
        self.W = self.W.to(labels.device)
        preds = F.softmax(preds, dim=1)
        labels = F.one_hot(labels, num_classes=self.n_classes).float()
        
        O = torch.matmul(labels.t(), preds)
        E = torch.matmul(labels.sum(dim=0).view(-1,1), preds.sum(dim=0).view(1,-1)) / O.sum()
        
        return (self.W*O).sum() / ((self.W*E).sum() + self.eps)