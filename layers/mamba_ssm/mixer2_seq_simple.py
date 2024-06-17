# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from functools import partial
import json
import os
import copy

from collections import namedtuple

import torch
import torch.nn as nn

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from mamba_ssm.modules.block import Block
from layers.mamba_ssm.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2

from utils.masking import random_shuffle, unshuffle
from einops import rearrange, reduce

from python_tsp.heuristics import solve_tsp_local_search, solve_tsp_simulated_annealing, \
    solve_tsp_lin_kernighan


def create_block(
        d_model,
        d_intermediate,
        ssm_cfg=None,
        # START
        n_vars=None,
        VPT_mode=0,
        dropout=0.,
        use_casual_conv=True,
        # END
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            dropout=dropout, use_casual_conv=use_casual_conv, n_vars=n_vars, VPT_mode=VPT_mode,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerTSModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            d_intermediate: int = 0,
            # vocab_size: int,
            ssm_cfg=None,
            n_vars=None,
            VPT_mode=0,
            ATSP_solver='SA',
            dropout=0.,
            use_casual_conv: bool = True,
            attn_layer_idx=None,
            attn_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.n_vars = n_vars

        self.VPT_mode = VPT_mode
        self.ATSP_solver = ATSP_solver
        print(f"VPT_mode: {VPT_mode}, ATSP_solver: {ATSP_solver}")

        if self.VPT_mode > 0:
            assert n_vars is not None, f"When VPT_mode ({self.VPT_mode}) > 0, n_vars should be passed in."
            self.d_model = d_model

        self.ids_shuffle = None

        # VAST: Variable-Aware Scan along Time
        self.adjacency_matrix = nn.Parameter(torch.zeros(n_vars, n_vars), requires_grad=False)
        self.count_matrix = nn.Parameter(torch.zeros(n_vars, n_vars), requires_grad=False)
        self.ending_points = nn.Parameter(torch.zeros(n_vars), requires_grad=False)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    # START
                    n_vars=self.n_vars,
                    VPT_mode=self.VPT_mode,
                    dropout=dropout,
                    use_casual_conv=use_casual_conv,
                    # END
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, x, inference_params=None, **mixer_kwargs):
        # TODO: flexible for n_vars
        n_vars_cur = self.n_vars
        if self.VPT_mode in [1]:
            if self.training:
                x = rearrange(x, 'b (c l) d -> b c (l d)', c=n_vars_cur)
                x, ids_shuffle, ids_restore = random_shuffle(x, mask_ratio=0, return_ids_shuffle=True)
                # Record the starting point
                ids_shuffle_ = torch.cat([ids_shuffle[:, [0]], ids_shuffle], dim=-1)
                self.transition_tuple = torch.stack([ids_shuffle_[:, :-1], ids_shuffle_[:, 1:]], dim=-1)

                x = rearrange(x, 'b c (l d) -> b (c l) d', d=self.d_model)
                self.ids_shuffle = None
            else:
                x = rearrange(x, 'b (c l) d -> b c (l d)', c=n_vars_cur)

                if self.ids_shuffle is None:
                    if self.ATSP_solver == 'GD':
                        adjacency_matrix = self.get_adjacency_matrix()

                        # Starting node
                        start_node = torch.argmin(torch.diag(adjacency_matrix)).item()

                        # Set diagonal elements to infinity to prevent self-loop
                        for j in range(adjacency_matrix.size(0)):
                            adjacency_matrix[j, j] = float("inf")
                        current_node = start_node
                        visited_nodes = [current_node]

                        # Traverse all nodes
                        while len(visited_nodes) < adjacency_matrix.size(0):
                            # Mark the current node as visited
                            adjacency_matrix[:, current_node] = float('inf')

                            # Find the nearest node to the current node
                            next_node = torch.argmin(adjacency_matrix[current_node]).item()

                            # Update current node to the next node
                            current_node = next_node
                            visited_nodes.append(current_node)

                        ids_shuffle = torch.tensor(visited_nodes, device=x.device)

                        self.ids_shuffle = ids_shuffle.repeat(x.shape[0], 1)
                    elif self.ATSP_solver in ['SA', 'LS', 'LK']:
                        adjacency_matrix = self.get_adjacency_matrix()

                        # # 起始节点
                        start_index = torch.argmin(torch.diag(adjacency_matrix)).item()
                        for j in range(adjacency_matrix.size(0)):
                            adjacency_matrix[j, j] = 0

                        distance_matrix = adjacency_matrix.cpu().numpy()
                        distance_matrix[:, start_index] = 0

                        # fix the 0-th bug in https://github.com/fillipe-gsm/python-tsp/issues/21
                        distance_matrix[:, [start_index, 0]] = distance_matrix[:, [0, start_index]]  # swap columns
                        distance_matrix[[start_index, 0], :] = distance_matrix[[0, start_index], :]  # swap rows

                        if self.ATSP_solver == 'SA':
                            shortest_path, distance = solve_tsp_simulated_annealing(distance_matrix,
                                                                                    x0=list(range(n_vars_cur)))
                        elif self.ATSP_solver == 'LS':
                            shortest_path, distance = solve_tsp_local_search(distance_matrix,
                                                                             x0=list(range(n_vars_cur)))
                        elif self.ATSP_solver == 'LK':
                            shortest_path, distance = solve_tsp_lin_kernighan(distance_matrix,
                                                                              x0=list(range(n_vars_cur)))
                        else:
                            raise NotImplementedError(f"Don't support the ATSP sovler {self.ATSP_solver}")

                        # recover the starting index
                        shortest_path[shortest_path.index(start_index)] = 0
                        shortest_path[0] = start_index
                        print(shortest_path, distance)

                        ids_shuffle = torch.tensor(shortest_path, device=x.device)
                        self.ids_shuffle = ids_shuffle.repeat(x.shape[0], 1)

                x, ids_restore = random_shuffle(x, ids_shuffle=self.ids_shuffle)
                x = rearrange(x, 'b c (l d) -> b (c l) d', d=self.d_model)
        else:
            ids_restore = None

        hidden_states = x
        residual = None

        ret_list = []
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )

        # input-level un-shuffle
        if self.VPT_mode in [1]:
            hidden_states = rearrange(hidden_states, 'b (c l) d -> b c (l d)', c=n_vars_cur)
            hidden_states = unshuffle(hidden_states, ids_restore)
            hidden_states = rearrange(hidden_states, 'b c (l d) -> b (c l) d', d=self.d_model)

        return hidden_states, ret_list

    def batch_update_state(self, cost_tensor):
        # print(cost_tensor.shape)
        cost_tensor = cost_tensor.detach()
        cost_tensor = reduce(cost_tensor, 'b t c -> b', 'mean')
        cost_tensor = cost_tensor - cost_tensor.mean()
        count_tensor = torch.ones_like(cost_tensor)

        B, C = cost_tensor.size(0), self.adjacency_matrix.size(0)

        # Calculate the indices of each coordinate in state_tensor
        indices = self.transition_tuple[:, :, 0] * C + self.transition_tuple[:, :, 1]

        # Scatter cost_tensor according to indices into state_tensor
        self.adjacency_matrix.view(-1).scatter_add_(0, indices.view(-1), cost_tensor.repeat(indices.shape[-1]))
        self.count_matrix.view(-1).scatter_add_(0, indices.view(-1), count_tensor.repeat(indices.shape[-1]))

        # Update the ending points
        self.ending_points.scatter_add_(0, self.transition_tuple[:, -1, 1], cost_tensor)

    def get_adjacency_matrix(self):
        adjacency_matrix = self.adjacency_matrix.clone() / self.count_matrix
        adjacency_matrix = adjacency_matrix + torch.abs(adjacency_matrix.min()) + 1e-7

        return adjacency_matrix

    def reset_ids_shuffle(self):
        self.ids_shuffle = None


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            d_intermediate: int,
            vocab_size: int,
            ssm_cfg=None,
            attn_layer_idx=None,
            attn_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states


class MambaLMHeadModel(nn.Module, GenerationMixin):

    def __init__(
            self,
            config: MambaConfig,
            initializer_cfg=None,
            device=None,
            dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        attn_layer_idx = config.attn_layer_idx
        attn_cfg = config.attn_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, inference_params=inference_params, **mixer_kwargs)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=4)
