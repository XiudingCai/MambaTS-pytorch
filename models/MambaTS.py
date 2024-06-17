import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange


class PredictionHead(nn.Module):
    def __init__(self, in_nf, out_nf):
        super().__init__()
        self.linear = nn.Linear(in_nf, out_nf)

    def forward(self, x, n_vars):
        x = rearrange(x, 'b (c l) d -> (b c) (l d)', c=n_vars)
        x = self.linear(x)
        x = rearrange(x, '(b c) p -> b p c', c=n_vars)

        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        # patching and embedding
        self.value_embedding = nn.Sequential(
            Rearrange('b c l d -> (b c) l d'),
            nn.Linear(configs.patch_len, configs.d_model, bias=False),
        )

        # Encoder
        from layers.mamba_ssm.mixer2_seq_simple import MixerTSModel as Mamba
        self.encoder = Mamba(
            d_model=configs.d_model,  # Model dimension d_model
            n_layer=configs.e_layers,
            n_vars=configs.enc_in,
            dropout=configs.dropout,
            ssm_cfg={'layer': 'Mamba1'},
            VPT_mode=configs.VPT_mode,
            ATSP_solver=configs.ATSP_solver,
            use_casual_conv=configs.use_casual_conv,
            fused_add_norm=True,
        )

        # Prediction Head
        self.head = PredictionHead(configs.d_model * configs.seq_len // configs.patch_len, configs.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        b, _, n_vars = x_enc.shape

        # Norm
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        # Do patching and embedding
        x = x_enc.permute(0, 2, 1)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        enc_in = self.value_embedding(x)

        # Variable Scan along Time (VST)
        enc_in = rearrange(enc_in, '(b c) l d -> b (c l) d', c=n_vars)

        # Encoder
        enc_out, attns = self.encoder(enc_in)

        # Decoder
        dec_out = self.head(enc_out, n_vars=n_vars)

        # De-norm
        dec_out = dec_out * (stdev[:, [0], :].repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, [0], :].repeat(1, self.pred_len, 1))

        return dec_out

    def batch_update_state(self, cost_tensor):
        self.encoder.batch_update_state(cost_tensor)

    def set_reordering_index(self, reordering_index):
        self.encoder.set_reordering_index(reordering_index)

    def reset_ids_shuffle(self):
        self.encoder.reset_ids_shuffle()
