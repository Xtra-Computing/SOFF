import gc
import torch

# from https://stackoverflow.com/questions/59265818/memory-leak-in-pytorch-object-detection


def debug_gpu():
    # Debug out of memory bugs.
    tensor_list = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or \
                    (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tensor_list.append(obj)
        except:
            pass
    return 'Count of tensors = {}.'.format(len(tensor_list))
