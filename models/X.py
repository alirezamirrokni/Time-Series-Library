import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from mamba_ssm import Mamba

from layers.Embed import DataEmbedding

class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len

        self.d_inner = configs.d_model * configs.expand
        self.dt_rank = math.ceil(configs.d_model / 16) # TODO implement "auto"
        
        self.embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.ttt = ttt(
                    vocab_size=32000,
                    hidden_size=128,
                    intermediate_size=5504,
                    num_hidden_layers=24,
                    num_attention_heads=32,
                    hidden_act="silu",
                    max_position_embeddings=2048,
                    initializer_range=0.02,
                    rms_norm_eps=1e-6,
                    use_cache=False,
                    pad_token_id=None,
                    bos_token_id=1,
                    eos_token_id=2,
                    pretraining_tp=1,
                    tie_word_embeddings=True,
                    rope_theta=10000.0,
                    use_gate=False,
                    share_qk=False,
                    ttt_layer_type="linear",
                    ttt_base_lr=1.0,
                    mini_batch_size_time=16,
                    mini_batch_size_variate=16,
                    pre_conv=False,
                    conv_kernel=4,
                    scan_checkpoint_group_size=0,
                    V = 100
        )

        self.out_layer = nn.Linear(configs.d_model, configs.c_out, bias=False)

    def forecast(self, x_enc, x_mark_enc):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        x = self.embedding(x_enc, x_mark_enc)
        x = self.mamba(x)
        x_out = self.out_layer(x)

        x_out = x_out * std_enc + mean_enc
        return x_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            x_out = self.forecast(x_enc, x_mark_enc)
            return x_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
