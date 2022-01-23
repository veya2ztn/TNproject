# -*- coding: utf-8 -*-
# CopyRight:
import random
import itertools

import numpy as np
import opt_einsum as oe

import torch
import torch.utils.benchmark as benchmark
from torch.testing._internal.common_utils import make_tensor
import random
import torch.backends.cudnn as cudnn
seed = 200
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class TestCase:
    def __init__(self, operands, sublists, out_sublist=None):
        self.operands = operands
        self.sublists = sublists
        self.out_sublist = out_sublist

    def sublist_format(self, explicit_output=True):
        args = [*itertools.chain(*zip(self.operands, self.sublists))]
        if explicit_output and self.out_sublist:
            args += self.out_sublist
        return args

    def equation_format(self, explicit_output=True):
        equation = ','.join(self._convert_sublist(sublist) for sublist in self.sublists)
        if explicit_output and self.out_sublist:
            equation += '->' + self._convert_sublist(self.out_sublist)
        return (equation, *self.operands)

    def _convert_label(self, label):
        if label == Ellipsis:
            return '...'
        elif label < 26:
            return chr(ord('a') + label)
        else:
            return chr(ord('A') + label - 26)

    def _convert_sublist(self, sublist):
        return ''.join(self._convert_label(label) for label in sublist)


def generate_random_test_cases(
        device, dtype,
        seed=None,
        n=10,
        n_labels=6,
        min_ops=1, max_ops=5,
        min_dims=1, max_dims=4,
        min_size=8, max_size=32,
        min_out_dim=1, max_out_dim=3,
        enable_diagonals=True,
        ellipsis_prob=0.25,
        broadcasting_prob=0.1):

    MAX_LABELS = 52

    assert 0 <= n
    assert 0 <= n_labels < MAX_LABELS
    assert 0 < min_ops <= max_ops
    assert 0 <= min_dims <= max_dims
    assert 0 <= min_size <= max_size
    assert 0 <= max_out_dim
    assert enable_diagonals or max_dims <= n_labels

    if seed:
        random.seed(seed)
        np.random.seed(seed)

    for _ in range(n):

        POSSIBLE_LABELS = np.random.choice(range(MAX_LABELS), n_labels, replace=False)
        LABELS_SIZE = np.random.randint(min_size, max_size + 1, MAX_LABELS)
        ELLIPSIS_SHAPE = np.random.randint(min_size, max_size + 1, max_dims - min_dims)

        operands = []
        sublists = []

        ell_size = 0
        valid_labels = set()

        for _ in range(random.randint(min_ops, max_ops)):
            n_dim = random.randint(min_dims, max_dims)
            labels = np.random.choice(POSSIBLE_LABELS, n_dim, replace=enable_diagonals)
            valid_labels.update(labels)
            shape = LABELS_SIZE[labels]

            mask = np.random.binomial(1, broadcasting_prob, n_dim)
            broadcast_labels = np.unique(labels[mask == 1])
            shape[np.isin(labels, broadcast_labels)] = 1

            labels = list(labels)
            shape = list(shape)

            if n_dim < max_dims and np.random.random() < ellipsis_prob:
                ell_num_dim = random.randint(1, max_dims - n_dim)
                ell_size = max(ell_size, ell_num_dim)
                ell_shape = ELLIPSIS_SHAPE[-ell_num_dim:]
                mask = np.random.binomial(1, broadcasting_prob, ell_num_dim)
                ell_shape[mask == 1] = 1
                ell_index = random.randint(0, n_dim)
                shape[ell_index:ell_index] = ell_shape
                labels.insert(ell_index, Ellipsis)

            operands.append(make_tensor(shape, device, dtype))
            sublists.append(labels)

        out_sublist = None
        if np.random.rand() < 0.5:
            num_out_labels = max(min_out_dim, random.randint(0, min(max_out_dim, len(valid_labels))) - ell_size)
            out_sublist = list(np.random.choice(list(valid_labels), num_out_labels, replace=False))
            out_sublist.insert(random.randint(0, num_out_labels), Ellipsis)

        yield TestCase(operands, sublists, out_sublist).equation_format()


def generate_test_cases_from_issue_57121(device, dtype):
    A = make_tensor((160,) * 2, device, dtype)
    B = make_tensor((160,) * 4, device, dtype)

    yield ('ij,ijkl->kl', A, B)
    yield ('kl,ijkl->ij', A, B)
    yield ('jk,ijkl->il', A, B)
    yield ('il,ijkl->jk', A, B)


def generate_test_cases_from_issue_32591(device, dtype):
    A = make_tensor((1, 1, 16, 2, 16, 2, 16, 2, 2, 2, 2), device, dtype)
    B = make_tensor((729, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2), device, dtype)

    yield ('...ijk,...ijk->...', A, B)

def generate_test_case_from_opt_einsum_example(device, dtype):
    dim = 10
    I = make_tensor((dim, dim, dim, dim), device, dtype)
    C = make_tensor((dim, dim), device, dtype)
    yield ('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)

test_case_generators = [
    generate_random_test_cases('cuda', torch.float, seed=2021, n=10, n_labels=8, min_ops=4, max_ops=6,
                               min_out_dim=4, max_out_dim=4, min_dims=2, max_size=16, enable_diagonals=False, broadcasting_prob=0)
]

results = []

for test_case_generator in test_case_generators:
    for equation, *operands in test_case_generator:
        print('equation:', equation, '\toperands:', [op.shape for op in operands])

        np_ops = [op.cpu().numpy() for op in operands]
        path = oe.contract_path(equation, *np_ops)[0]

        out=torch.einsum(equation, *operands);print(out.norm().item())
        out=torch.einsum(equation, *operands, optimize=path);print(out.norm().item())
        out=   np.einsum(equation, *np_ops, optimize=('einsum_path', *path));print(np.linalg.norm(out))
        # results.append(benchmark.Timer(
        #     stmt='out=torch.einsum(equation, *operands);print(out.norm().item())',
        #     globals={'equation': equation, 'operands': operands},
        #     sub_label=equation,
        #     description='torch.einsum',
        # ).blocked_autorange())
        #
        # results.append(benchmark.Timer(
        #     stmt='out=torch.einsum(equation, *operands, optimize=optimize);print(out.norm().item())',
        #     globals={'equation': equation, 'operands': operands, 'optimize': path},
        #     sub_label=equation,
        #     description='torch.einsum opt',
        # ).blocked_autorange())
        #
        # results.append(benchmark.Timer(
        #     stmt='out=numpy.einsum(equation, *operands, optimize=optimize);print(out.norm().item())',
        #     setup='import numpy',
        #     globals={'equation': equation, 'operands': np_ops, 'optimize': ('einsum_path', *path)},
        #     sub_label=equation,
        #     description='numpy.einsum opt',
        # ).blocked_autorange())

# compare = benchmark.Compare(results)
# compare.trim_significant_figures()
# compare.colorize(rowwise=True)
# compare.print()
