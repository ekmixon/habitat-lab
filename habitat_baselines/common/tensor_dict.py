#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy
import numbers
from typing import Callable, Dict, Optional, Tuple, Union, overload

import numpy as np
import torch

TensorLike = Union[torch.Tensor, np.ndarray, numbers.Real]
DictTree = Dict[str, Union[TensorLike, "DictTree"]]
TensorIndexType = Union[int, slice, Tuple[Union[int, slice], ...]]


class TensorDict(Dict[str, Union["TensorDict", torch.Tensor]]):
    r"""A dictionary of tensors that can be indexed like a tensor or like a dictionary.

    .. code:: py
        t = TensorDict(a=torch.randn(2, 2), b=TensorDict(c=torch.randn(3, 3)))

        print(t)

        print(t[0, 0])

        print(t["a"])

    """

    @classmethod
    def from_tree(cls, tree: DictTree) -> "TensorDict":
        res = cls()
        for k, v in tree.items():
            res[k] = cls.from_tree(v) if isinstance(v, dict) else torch.as_tensor(v)
        return res

    def to_tree(self) -> DictTree:
        return {
            k: v.to_tree() if isinstance(v, TensorDict) else v
            for k, v in self.items()
        }

    @overload
    def __getitem__(self, index: str) -> Union["TensorDict", torch.Tensor]:
        ...

    @overload
    def __getitem__(self, index: TensorIndexType) -> "TensorDict":
        ...

    def __getitem__(
        self, index: Union[str, TensorIndexType]
    ) -> Union["TensorDict", torch.Tensor]:
        if isinstance(index, str):
            return super().__getitem__(index)
        else:
            return TensorDict({k: v[index] for k, v in self.items()})

    @overload
    def set(
        self,
        index: str,
        value: Union[TensorLike, "TensorDict", DictTree],
        strict: bool = True,
    ) -> None:
        ...

    @overload
    def set(
        self,
        index: TensorIndexType,
        value: Union["TensorDict", DictTree],
        strict: bool = True,
    ) -> None:
        ...

    def set(
        self,
        index: Union[str, TensorIndexType],
        value: Union[TensorLike, "TensorDict"],
        strict: bool = True,
    ) -> None:
        if isinstance(index, str):
            super().__setitem__(index, value)
        else:
            if strict and (self.keys() != value.keys()):
                raise KeyError(f"Keys don't match: Dest={self.keys()} Source={value.keys()}")

            for k in self.keys():
                if k not in value:
                    if strict:
                        raise KeyError(f"Key {k} not in new value dictionary")
                    else:
                        continue

                v = value[k]

                if isinstance(v, (TensorDict, dict)):
                    self[k].set(index, v, strict=strict)
                else:
                    self[k][index].copy_(torch.as_tensor(v))

    def __setitem__(
        self,
        index: Union[str, TensorIndexType],
        value: Union[torch.Tensor, "TensorDict"],
    ):
        self.set(index, value)

    @classmethod
    def map_func(
        cls,
        func: Callable[[torch.Tensor], torch.Tensor],
        src: "TensorDict",
        dst: Optional["TensorDict"] = None,
    ) -> "TensorDict":
        if dst is None:
            dst = TensorDict()

        for k, v in src.items():
            if torch.is_tensor(v):
                dst[k] = func(v)
            else:
                dst[k] = cls.map_func(func, v, dst.get(k, None))

        return dst

    def map(
        self, func: Callable[[torch.Tensor], torch.Tensor]
    ) -> "TensorDict":
        return self.map_func(func, self)

    def map_in_place(
        self, func: Callable[[torch.Tensor], torch.Tensor]
    ) -> "TensorDict":
        return self.map_func(func, self, self)

    def __deepcopy__(self, _memo=None) -> "TensorDict":
        return TensorDict.from_tree(copy.deepcopy(self.to_tree(), memo=_memo))
