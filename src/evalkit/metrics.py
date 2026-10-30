### cross entropy, perplexity and more
import torch
import torch.nn as nn

def cross_entropy_loss(logits, targets):
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, targets)
    return loss

def perplexity_n(loss):
    perplexity = torch.exp(loss)
    return perplexity.item() # cli output format 


