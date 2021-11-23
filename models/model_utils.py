import torch

def get_chain_contraction(tensor):
    size   = int(tensor.shape[0])
    while size > 1:
        half_size = size // 2
        nice_size = 2 * half_size
        leftover  = tensor[nice_size:]
        tensor    = torch.einsum("mbik,mbkj->mbij",tensor[0:nice_size:2], tensor[1:nice_size:2])
        #(k/2,NB,D,D),(k/2,NB,D,D) <-> (k/2,NB,D,D)
        tensor   = torch.cat([tensor, leftover], axis=0)
        size     = half_size + int(size % 2 == 1)
    return tensor.squeeze(0)
