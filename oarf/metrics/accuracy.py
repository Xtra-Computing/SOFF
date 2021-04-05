import torch
from torch import nn
from functools import partial


def binary(output, ground_truth) -> float:
    rounded = torch.round(output)
    num_correct = torch.sum((rounded == ground_truth.int()))
    acc = num_correct / len(ground_truth)
    return acc


def topk(output, ground_truth, k=1) -> float:
    batch_size = ground_truth.size(0)

    _, pred = output.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(ground_truth.view(1, -1).expand_as(pred))

    correct_k = correct[:k].reshape(-1).float().sum(0)
    return correct_k.mul_(100.0 / batch_size)


def top1(output, ground_truth) -> float:
    return topk(output, ground_truth, 1)


def top5(output, ground_truth) -> float:
    return topk(output, ground_truth, 5)

