import numpy as np
import math
from random import randrange, choice, choices
import itertools
import torch

# Adding extra padding is an easy way to improve performance, as it gives the
# model more space to think. For example, without padding, the standard model
# kind=transformer-lstm gets accuracies (1, 0.95, 0.66), but we just five extra
# paddings on the left, it gets (1, 0.98, 0.80). Even better, if we add the
# padding right before the equality sign, ...


class Dataset:
    def __init__(self, base, number_length, pre_end_padding=0, flip=False, preferred_dtype='int64'):
        self.base = base
        self.number_length = number_length
        self.pre_end_padding = pre_end_padding
        self.flip = flip
        self.preferred_dtype = preferred_dtype

        self.start_token = base  # Before input
        self.end_token = base + 1  # After input
        self.separator_token = base + 2  # between inputs
        self.padding_token = base + 3  # Before input and after target
        self.eos_token = base + 4  # After target
        self.n_tokens = base + 5

        self.dic = {i: str(i) for i in range(self.base + 1)}
        self.dic[self.padding_token] = ""
        self.dic[self.start_token] = ""
        self.dic[self.end_token] = "="
        self.dic[self.eos_token] = ""

    def make_numbers(self, shape, number_length=None):
        if number_length is None:
            number_length = self.number_length
        if np.dtype(self.preferred_dtype) is np.dtype(object):
            powers = [self.base**(i+1) for i in range(number_length)]
            n = math.prod(shape)
            result = np.array([randrange(choice(powers)) for i in range(n)],
                              dtype=object).reshape(shape)
        else:
            digits_shape = shape + (number_length,)
            digits = np.random.randint(0, self.base, math.prod(digits_shape)).reshape(digits_shape)
            # n_digits = np.random.randint(0, number_length, shape)
            # mask = np.arange(number_length) < n_digits[..., None]
            exponents = np.arange(number_length - 1, -1, -1, dtype=self.preferred_dtype)
            bases = np.expand_dims(np.power(self.base, exponents), 0)
            result = (digits * bases).sum(axis=-1)
        return result

        # if number_length is None:
        #     number_length = self.number_length
        # first_digit = torch.randint(1, self.base, shape + (1,))
        # other_digits = torch.randint(self.base, shape + (number_length-1,))
        # digits = torch.cat((first_digit, other_digits), dim=-1 )
        # # # Evenly distributing numbers to be of different length upto number_length
        # # n_digits = torch.randint(number_length, shape)
        # # mask = torch.arange(number_length) < n_digits[..., None]
        # # digits[mask] = 0
        # bases = torch.pow(self.base, torch.arange(number_length - 1, -1, -1)).unsqueeze(
        #     0
        # )
        # return (digits * bases).sum(dim=-1)
    
    def to_digits(self, numbers, length=None):
        if length is None:
            length = self.number_length

        # Convert numbers to digits
        tensor = np.tile(np.expand_dims(numbers, 1), (1, length))
        exponents = np.arange(length - 1, -1, -1, dtype=self.preferred_dtype)
        bases = np.expand_dims(np.power(self.base, exponents), 0)
        digits = (tensor // bases) % self.base

        # Mask leading zeros
        mask = digits.cumsum(1) == 0
        mask[:, -1] = False
        digits[mask] = self.padding_token
        if self.flip:
            return np.flip(digits, [1])
        return digits
    
        # # Convert numbers to digits
        # tensor = numbers.unsqueeze(1).repeat(1, length)
        # bases = torch.pow(
        #     self.base, torch.arange(length - 1, -1, -1, device=tensor.device)
        # ).unsqueeze(0)
        # digits = (tensor // bases) % self.base

        # # Mask leading zeros
        # mask = digits.cumsum(1) == 0
        # mask[:, -1] = False
        # digits[mask] = self.padding_token
        # if self.flip:
        #     return torch.flip(digits, [1])
        # return digits

    def move_padding_to_end(self, tensor, end=True):
        """Move all padding tokens in each row to the end without reordering the rest."""

        # Create a tensor with large values where there's padding and row-wise indices elsewhere
        # This allows us to "sort" the padding to the end, while keeping everything else in its
        # original order.
        sorting_tensor = np.where(
            tensor == self.padding_token,
            tensor.shape[1] if end else -tensor.shape[1],
            np.arange(tensor.shape[1])
        )

        # Get the indices that would sort the tensor
        sorted_indices = np.argsort(sorting_tensor, axis=1)

        # Use the sorted indices to rearrange the original tensor
        sorted_tensor = np.take_along_axis(tensor, sorted_indices, 1)

        return sorted_tensor
    
        # sorting_tensor = torch.where(
        #     tensor == self.padding_token,
        #     tensor.size(1) if end else -tensor.size(1),
        #     torch.arange(tensor.size(1), device=tensor.device),
        # )

        # # Get the indices that would sort the tensor
        # _, sorted_indices = sorting_tensor.sort(dim=1)

        # # Use the sorted indices to rearrange the original tensor
        # sorted_tensor = torch.gather(tensor, 1, sorted_indices)

        # return sorted_tensor

    def generate_batch(self, bs):
        res, carry = self._generate_batch(bs)
        res = self.move_padding_to_end(res)

        # Insert COT padding
        if self.pre_end_padding != 0:
            indices_padding = (res == self.end_token).nonzero(as_tuple=True)
            expanded_tensor = np.zeros(bs, self.seq + self.pre_end_padding, dtype=res.dtype)
            # expanded_tensor = torch.zeros(bs, self.seq + self.pre_end_padding, dtype=res.dtype)
            # Calculate the positions in the expanded tensor for all elements
            positions = np.tile(np.expand_dims(np.arange(self.seq), 0), (bs, 1))
            # positions = torch.arange(self.seq).unsqueeze(0).repeat(bs, 1)
            positions += self.pre_end_padding * (positions >= indices_padding[1].unsqueeze(1))
            # Use scatter to insert values at the correct positions
            expanded_tensor.scatter_(1, positions, res)
            res = expanded_tensor

        # assert res.shape == (bs, self.seq)
        return res, carry.tolist()

    def _generate_batch(self, tokens):
        assert False, "Not implemented"

    def repr_example(self, example):
        tokens = [
            (tuple(group)[::-1] if self.flip else tuple(group))
            if is_number
            else next(group)
            for is_number, group in itertools.groupby(
                example.tolist(), key=lambda x: x < self.base
            )
        ]
        return self._repr_tokens(tokens).strip()

    def _repr_tokens(self, tokens):
        res = []
        for token in tokens:
            if type(token) is tuple:
                res.append("".join(map(str, token)))
            else:
                res.append(self.dic[token])
        return " ".join(res)

    @property
    def seq(self):
        assert False, "Not implemented"


class BinaryOpDataset(Dataset):
    def __init__(
        self,
        base,
        number_length,
        func,
        sep,
        out_length,
        pre_end_padding=0,
        min_b=0,
        flip=False,
        **kwargs,
    ):
        super().__init__(base, number_length, pre_end_padding, flip, **kwargs)
        self.func = func
        self.sep_string = sep
        self.out_length = out_length
        self.dic[self.separator_token] = sep
        self.min_b = min_b

    def _generate_batch(self, bs):
        a, b = self.make_numbers((2, bs))
        b = np.clip(b, self.min_b, None)
        out = self.func(a, b)
        A = self.to_digits(a)
        B = self.to_digits(b)
        C = np.zeros((bs, self.number_length+1))
        carry = np.zeros((bs, self.number_length+1, self.number_length+1))
        Ap = np.where(A!=13, A, 0)
        Bp = np.where(B!=13, B, 0)
        for n in range(self.number_length):
            mask = Ap[:,n] + Bp[:,n] + C[:,n] >= self.base
            C[:,n+1][mask]=1
            carry[:,n+1] = C
        return np.concatenate(
            [
                np.full((bs, 1), self.start_token),
                A,
                np.full((bs, 1), self.separator_token),
                B,
                np.full((bs, 1), self.end_token),
                self.to_digits(out, length=self.out_length),
                np.full((bs, 1), self.eos_token),
            ],
            axis=1,
        ), carry

    @property
    def seq(self):
        return self.number_length * 2 + self.out_length + 4
    