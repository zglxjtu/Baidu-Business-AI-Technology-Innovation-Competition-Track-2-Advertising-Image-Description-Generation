# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import partial
from typing import Dict, List

import numpy as np
import paddle
import requests
import paddle.nn as nn
from paddle.vision.transforms import functional as F
from paddlenlp.utils.initializer import normal_, xavier_uniform_, constant_
from PIL import Image

from .vit import VisionTransformer, VisionTransformerConfig, get_abs_pos


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class MultiHeadAttention(nn.Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.

    Please refer to `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_
    for more details.

    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        num_heads (int): The number of heads in multi-head attention.
        dropout (float, optional): The dropout probability used on attention
            weights to drop some attention targets. 0 for no dropout. Default 0
        kdim (int, optional): The feature size in key. If None, assumed equal to
            `embed_dim`. Default None.
        vdim (int, optional): The feature size in value. If None, assumed equal to
            `embed_dim`. Default None.
        need_weights (bool, optional): Indicate whether to return the attention
            weights. Default False.

    Examples:

        .. code-block:: python

            import paddle

            # encoder input: [batch_size, sequence_length, d_model]
            query = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, num_heads, query_len, query_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            multi_head_attn = paddle.nn.MultiHeadAttention(128, 2)
            output = multi_head_attn(query, None, None, attn_mask=attn_mask)  # [2, 4, 128]
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        kdim=None,
        vdim=None,
        need_weights=False,
    ):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim:
            self.in_proj_weight = self.create_parameter(
                shape=[embed_dim, 3 * embed_dim],
                attr=None,
                dtype=self._dtype,
                is_bias=False,
            )
            self.in_proj_bias = self.create_parameter(
                shape=[3 * embed_dim], attr=None, dtype=self._dtype, is_bias=True
            )
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(self.kdim, embed_dim)
            self.v_proj = nn.Linear(self.vdim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._type_list = ("q_proj", "k_proj", "v_proj")

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            else:
                constant_(p)

    def compute_qkv(self, tensor, index):
        if self._qkv_same_embed_dim:
            tensor = paddle.nn.functional.linear(
                x=tensor,
                weight=paddle.cast(
                    self.in_proj_weight[:, index * self.embed_dim : (index + 1) * self.embed_dim], tensor.dtype
                ),
                bias=paddle.cast(
                    self.in_proj_bias[index * self.embed_dim : (index + 1) * self.embed_dim], tensor.dtype
                )
                if self.in_proj_bias is not None
                else None,
            )
        else:
            tensor = getattr(self, self._type_list[index])(tensor)
        tensor = tensor.reshape([0, 0, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        return tensor

    def forward(self, query, key=None, value=None, attn_mask=None):
        r"""
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.

        Parameters:
            query (Tensor): The queries for multi-head attention. It is a
                tensor with shape `[batch_size, query_length, embed_dim]`. The
                data type should be float32 or float64.
            key (Tensor, optional): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, key_length, kdim]`. The
                data type should be float32 or float64. If None, use `query` as
                `key`. Default None.
            value (Tensor, optional): The values for multi-head attention. It
                is a tensor with shape `[batch_size, value_length, vdim]`.
                The data type should be float32 or float64. If None, use `query` as
                `value`. Default None.
            attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False`
                values and the others have `True` values. When the data type is
                int, the unwanted positions have 0 values and the others have 1
                values. When the data type is float, the unwanted positions have
                `-INF` values and the others have 0 values. It can be None when
                nothing wanted or needed to be prevented attention to. Default None.

        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `query`, representing attention output. Or a tuple if \
                `need_weights` is True or `cache` is not None. If `need_weights` \
                is True, except for attention output, the tuple also includes \
                the attention weights tensor shaped `[batch_size, num_heads, query_length, key_length]`. \
                If `cache` is not None, the tuple then includes the new cache \
                having the same type as `cache`, and if it is `StaticCache`, it \
                is same as the input `cache`, if it is `Cache`, the new cache \
                reserves tensors concatanating raw tensors with intermediate \
                results of current query.
        """
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        q, k, v = (self.compute_qkv(t, i) for i, t in enumerate([query, key, value]))

        # scale dot product attention
        product = paddle.matmul(x=q, y=k, transpose_y=True)
        scaling = float(self.head_dim) ** -0.5
        product = product * scaling

        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask
        weights = paddle.nn.functional.softmax(product)
        if self.dropout:
            weights = paddle.nn.functional.dropout(weights, self.dropout, training=self.training, mode="upscale_in_train")

        out = paddle.matmul(weights, v)

        # combine heads
        out = paddle.transpose(out, perm=[0, 2, 1, 3])
        out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        return out if len(outs) == 1 else tuple(outs)


class Resampler(paddle.nn.Layer):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(self, grid_size, embed_dim, num_heads, kv_dim=None, norm_layer=paddle.nn.LayerNorm):
        super().__init__()
        self.num_queries = grid_size**2
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        x = paddle.to_tensor(get_2d_sincos_pos_embed(embed_dim, grid_size), dtype=paddle.get_default_dtype())
        self.pos_embed = paddle.create_parameter(
            shape=x.shape, dtype=x.dtype, default_initializer=paddle.nn.initializer.Assign(x)
        )

        x = paddle.zeros(shape=[self.num_queries, embed_dim], dtype=paddle.get_default_dtype())
        self.query = paddle.create_parameter(
            shape=x.shape, dtype=x.dtype, default_initializer=paddle.nn.initializer.Assign(x)
        )
        self.query.stop_gradient = True

        normal_(self.query, mean=0.0, std=0.02)
        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = paddle.nn.Linear(in_features=kv_dim, out_features=embed_dim, bias_attr=False)
        else:
            self.kv_proj = paddle.nn.Identity()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            normal_(m.weight, std=0.02)
            if isinstance(m, paddle.nn.Linear) and m.bias is not None:
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(m.bias)
        elif isinstance(m, paddle.nn.LayerNorm):
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(m.bias)
            init_Constant = paddle.nn.initializer.Constant(value=1.0)
            init_Constant(m.weight)

    def forward(self, x, attn_mask=None):
        pos_embed = get_abs_pos(self.pos_embed, x.shape[1])
        x = self.kv_proj(x)

        x = self.ln_kv(x).transpose(perm=[1, 0, 2])
        N = x.shape[1]

        q = self.ln_q(self.query)

        query = (self._repeat(q, N) + self.pos_embed.unsqueeze(axis=1)).transpose(perm=[1, 0, 2])
        key = (x + pos_embed.unsqueeze(axis=1)).transpose(perm=[1, 0, 2])
        value = x.transpose(perm=[1, 0, 2])

        out = self.attn(query, key, value, attn_mask=attn_mask)

        return out

    def _repeat(self, query, N: int):
        return paddle.tile(query.unsqueeze(axis=1), [1, N, 1])


class Vision(paddle.nn.Layer):
    def __init__(
        self,
        config: Dict[str, int],
        n_queries: int = 256,
    ):
        super().__init__()
        image_height, image_width = self.image_size = config["image_size"], config["image_size"]
        patch_height, patch_width = self.patch_size = config["patch_size"], config["patch_size"]
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.output_dim = config["output_dim"]

        vit_config = VisionTransformerConfig(**config)
        self.vit = VisionTransformer(vit_config)
        norm_layer = partial(paddle.nn.LayerNorm, epsilon=1e-06)

        self.attn_pool = Resampler(
            grid_size=int(math.sqrt(n_queries)),
            embed_dim=self.output_dim,
            num_heads=self.output_dim // 128,
            kv_dim=config["width"],
            norm_layer=norm_layer,
        )
        self.ln_post = norm_layer(self.output_dim)
        self.proj = paddle.create_parameter(
            shape=[self.output_dim, self.output_dim],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Assign(
                self.output_dim**-0.5 * paddle.randn(shape=[self.output_dim, self.output_dim])
            ),
        )
        self.proj.stop_gradient = True

    def image_transform(self, image):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]

        image = F.resize(image, size=self.image_size, interpolation="bicubic")
        tensor_normalize = paddle.vision.transforms.Normalize(mean=mean, std=std, data_format="HWC")
        image = tensor_normalize(np.array(image) / 255.0)
        image = F.to_tensor(image)

        return image

    def forward(self, x: paddle.Tensor):
        x = self.vit(x)

        x = self.attn_pool(x)

        x = self.ln_post(x)
        x = x @ self.proj

        return x

    def prepare(self, image_paths: List[str]):
        images = []
        for image_path in image_paths:
            if image_path.startswith("http://") or image_path.startswith("https://"):
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                image = Image.open(image_path)
            image = image.convert("RGB")
            images.append(self.image_transform(image))
        images = paddle.stack(x=images, axis=0)
        return images
