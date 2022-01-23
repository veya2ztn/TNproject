### The opt_einsum version PyTorch
The PyTorch has only normal version `einsum`, which is usually not optimized.
To get a optimized contraction path for large tensor network, the valid way is turning to [opt_einsum](https://github.com/dgasmith/opt_einsum).

After getting the optimized path, there are two way to assign contraction task:

- via opt_einsum build-in engine, `oe.contract(equation,*operands, optimize=path)`
- via a optimized einsum [custom PyTorch version](https://github.com/PyTorch/PyTorch/tree/heitorschueroff/einsum%2Foptimize) (OECPyTorch)

They have slightly difference: For the `oe.contract`, it contract operation one by one. On the other hand, the PyTorch will expand all tensor to high but redundancy dimension(see [here](https://github.com/PyTorch/PyTorch/blob/master/aten/src/ATen/native/Linear.cpp) for more detail)，then make `sum` operation together. The OECPyTorch will switch the `sum` order via given argument `path`, to achieve fast contraction. This two programming way is mathematically same but only with realization difference.

--------

After several tests, they **have same performance**.

![](https://github.com/veya2ztn/TNproject/blob/main/benchmark/ContractionEngine/figures/Memory_comparsion_between_complie_method_and_oe_method_for_different_batch.png)

![](https://github.com/veya2ztn/TNproject/blob/main/benchmark/ContractionEngine/figures/Speed_comparsion_between_complie_method_and_oe_method_for_different_batch.png)

But I will recommend you use the build-in function `oe.contract` because the custom PyTorch way has following backward:

1. To achieve this modified branch, we need compile this version torch from the beginning.

2. The PyTorch can only handle alphabet(`a-zA-z`) as string input (although [PyTorch 1.10](https://PyTorch.org/docs/stable/generated/torch.einsum.html)  support sublist  format, it still convert sublist to a-zA-z). So the number of tensor has a limitation up to 52=26+26.

3. Meanwhile, the CUDA, which is the GPU backend of PyTorch, only support  64 dims tensor. As introduced above, the PyTorch einsum will expand all tensors to very large dimension, so the upper contraction bond is 64.

To overcome problem 1, we can package the `whl` binary package, but it is old version PyTorch, so lose some newest feature.

To overcome problem  2, one solution is  change the C++ function `Liner.cpp:einsum`'s argument from  `string` to `u32string`, but it will result much more changes on other file. To avoid change the type from `string` to `u32string`, one idea is transfer the`letter string` "...ab,...bc,...cd -> ...ad" to `number string` like "...<0><1>,...<1><2>,...<2><3> -> ...<0><3>". But remember, such a coding way is not serious, so I won't pull-request to PyTorch.

We only need change 2 places.

- ===> `pytorch/aten/src/ATen/native/Linear.cpp`.
  For example, to identity the `label`, need an extra string buffer

  ```
     case '<':
          num_string="";
          break;
        case '>':
          TORCH_CHECK(
              to_label(num_string)<TOTAL_LABELS,
              "einsum(): expected subscripts to be small than",TOTAL_LABELS,
              "but got ",num_string
            );
          op_labels.back().push_back(to_label(num_string));
          ++label_count[to_label(num_string)];
          break;
        default:
          num_string+=string(1,*it);
          break;
  ```

- ===> `pytorch/torch/functional.py`
  The newest pytorch support `sublist` as input, which is great.
  Such supporting is realized by converting `sublist` to `letter string`, the limitation still in [a-zA-Z].
  However, if we use `number string` idea, the limit is the size of unit8_t (255).
  And we can do `alpha_canonicalize` like in https://github.com/dgasmith/opt_einsum/blob/master/opt_einsum/parser.py.
  So it is potential to support any letter include Greek
  (Unless the number of used letters smaller than 255)

To overcome problem 3, however there is no way to achieve except `maybe` compile CUDA.
When test on contration which involed more than 64 bonds , it appear

```
return _VF.einsum(equation, operands, optimize=optimize)  # type: ignore[attr-defined]
RuntimeError: only tensors with up to 64 dims are supported
```

Based on these, I recommend to use the build-in `oe.contract` in `opt_einsum`
