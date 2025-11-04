"""_summary_
forecast_net is designed to use a variety of information sources to forcast trends in the
stock market over a 7 (trading) day period. 


financial info we will be using:
--------------------------------
90 day return history of the stock (close value - open value) represented as a tensor (B, 90, 1)

90 day volume history (volume for each of the previous 90 days ligned up with the above) (B, 90, 1)

90 day market value history (related to the return history, 
but can let us know if there were any stock splits which can impact the forecast) (B, 90, 1)

yahoo finance's calculation of the most recent beta value to express the volatility of the stock (B, 1, 1)

90 day history of the P/E value used to calculate how the stock moves in general (B, 90, 1)


"""

import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 1024, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, L, d_model)
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1)]
    
class ForecastNetPositionalEncoder(nn.Module):
    def __init__(self, d_model:int = 1024, max_len:int = 10000, num_modalities:int = 5):
        super().__init__()
        # sinusoidal time encoding
        self.sinusoidal_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
        # learnable bias over time positions
        self.learnable_bias = nn.Parameter(torch.zeros(1, max_len, d_model))
        # modality embeddings
        self.modality_embed = nn.Embedding(num_modalities, d_model)

    def forward(self, x: torch.Tensor, modality_ids: torch.Tensor):
        """
        Args:
            x: (B, L, d_model)
            modality_ids: (B, L) long tensor of modality indices
        """
        time_encoded = self.sinusoidal_encoding(x) + self.learnable_bias[:, :x.size(1)]
        modality_encoded = self.modality_embed(modality_ids)  # (B, L, d_model)
        return time_encoded + modality_encoded


class ResidualInputProj(nn.Module):
    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.residual_proj = (
            nn.Linear(in_dim, d_model) if in_dim != d_model else nn.Identity()
        )
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x: torch.Tensor):
        proj = self.proj(x)
        res = self.residual_proj(x)
        return self.norm(proj + res)



class ForecastNetEncoder(nn.Module):
    def __init__(self, d_model=1024, in_dim=1,
                 nhead_enc=8, num_layers=16, max_len=10000,
                 pe_as_sequence: bool = False):  # <— new toggle
        super().__init__()
        
        self.pe_as_sequence = pe_as_sequence

        self.val_ts_proj = ResidualInputProj(in_dim, d_model)
        self.vol_ts_proj = ResidualInputProj(in_dim, d_model)
        self.mval_ts_proj = ResidualInputProj(in_dim, d_model)

        # keep PE projection only if you want to use it as a sequence
        if self.pe_as_sequence:
            self.pe_ts_proj = ResidualInputProj(in_dim, d_model)

        # FiLM conditioning for beta (global scalar) — already added
        self.beta_gamma = nn.Linear(in_dim, d_model)
        self.beta_beta  = nn.Linear(in_dim, d_model)

        # FiLM conditioning for PE (from pooled sequence)
        self.pe_gamma   = nn.Linear(in_dim, d_model)
        self.pe_beta    = nn.Linear(in_dim, d_model)

        # modality-aware positional encoder
        num_modalities = 4 if self.pe_as_sequence else 3
        self.pos_encoder = ForecastNetPositionalEncoder(
            d_model=d_model, max_len=max_len, num_modalities=num_modalities
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead_enc, batch_first=True
        )
        self.forecast_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, val_ts, vol_ts, mval_ts, beta, pe_ts):
        B = val_ts.size(0)

        val_ts_proj = self.val_ts_proj(val_ts)     # (B, 90, d_model)
        vol_ts_proj = self.vol_ts_proj(vol_ts)     # (B, 90, d_model)
        mval_ts_proj = self.mval_ts_proj(mval_ts)  # (B, 90, d_model)

        if self.pe_as_sequence:
            pe_ts_proj  = self.pe_ts_proj(pe_ts)   # (B, 90, d_model)
            seq = torch.cat((val_ts_proj, vol_ts_proj, mval_ts_proj, pe_ts_proj), dim=1)  # (B, 360, d_model)
            modality_ids = torch.cat((
                torch.full((B, 90), 0, dtype=torch.long, device=val_ts.device),
                torch.full((B, 90), 1, dtype=torch.long, device=val_ts.device),
                torch.full((B, 90), 2, dtype=torch.long, device=val_ts.device),
                torch.full((B, 90), 3, dtype=torch.long, device=val_ts.device),
            ), dim=1)
        else:
            # no PE tokens — only 3 streams (val/vol/mval)
            seq = torch.cat((val_ts_proj, vol_ts_proj, mval_ts_proj), dim=1)  # (B, 270, d_model)
            modality_ids = torch.cat((
                torch.full((B, 90), 0, dtype=torch.long, device=val_ts.device),
                torch.full((B, 90), 1, dtype=torch.long, device=val_ts.device),
                torch.full((B, 90), 2, dtype=torch.long, device=val_ts.device),
            ), dim=1)

        # time + modality encodings
        seq = self.pos_encoder(seq, modality_ids)

        # ----- FiLM conditioning (β + PE) -----
        # beta: (B,1,1) -> Linear -> (B,1,d_model)
        gamma_b = self.beta_gamma(beta)  # (B,1,d_model)
        beta_b  = self.beta_beta(beta)   # (B,1,d_model)

        # P/E: pool sequence to snapshot then Linear
        pe_snap = pe_ts.mean(dim=1, keepdim=True)   # (B,1,1)
        gamma_p = self.pe_gamma(pe_snap)            # (B,1,d_model)
        beta_p  = self.pe_beta(pe_snap)             # (B,1,d_model)

        gamma = gamma_b + gamma_p                   # (B,1,d_model)
        beta_  = beta_b  + beta_p                   # (B,1,d_model)

        # broadcast over L: (B,L,d_model) * (B,1,d_model) + (B,1,d_model)
        seq = seq * (1 + gamma) + beta_
# --------------------------------------

        # --------------------------------------

        encoded = self.forecast_encoder(seq)
        return self.norm(encoded)



class ForecastNetDecoder(nn.Module):
    def __init__(self, d_model=1024, nhead_dec=8, num_layers_dec=16, query_len=7, in_dim=1):
        super().__init__()
        self.query_len = query_len
        self.query_embed = nn.Parameter(torch.randn(1, query_len, d_model))

        # FiLM for beta & PE
        self.beta_gamma = nn.Linear(in_dim, d_model)
        self.beta_beta  = nn.Linear(in_dim, d_model)
        self.pe_gamma   = nn.Linear(in_dim, d_model)
        self.pe_beta    = nn.Linear(in_dim, d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead_dec, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers_dec)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, memory, beta, pe_ts):
        B = memory.size(0)
        query = self.query_embed.expand(B, -1, -1)  # (B,7,d_model)

        # β FiLM
        gamma_b = self.beta_gamma(beta)  # (B,1,d_model)
        beta_b  = self.beta_beta(beta)   # (B,1,d_model)

        # P/E FiLM (pool to snapshot)
        pe_snap = pe_ts.mean(dim=1, keepdim=True)  # (B,1,1)
        gamma_p = self.pe_gamma(pe_snap)           # (B,1,d_model)
        beta_p  = self.pe_beta(pe_snap)            # (B,1,d_model)

        gamma = gamma_b + gamma_p                  # (B,1,d_model)
        beta_  = beta_b  + beta_p                  # (B,1,d_model)

        # query: (B,7,d_model) — broadcast FiLM over its length
        query = query * (1 + gamma) + beta_


        decoded = self.decoder(tgt=query, memory=memory)
        return self.norm(decoded)

    
    
class ForecastNet(nn.Module):
    def __init__(self, in_dim=1, d_model=1024,
                 n_head_enc=8, n_head_dec=8,
                 num_layers_enc=16, num_layers_dec=16,
                 query_len=7, max_len=10000,
                 pe_as_sequence: bool = False):
        super().__init__()

        self.encoder = ForecastNetEncoder(
            d_model=d_model, in_dim=in_dim,
            nhead_enc=n_head_enc, num_layers=num_layers_enc,
            max_len=max_len, pe_as_sequence=pe_as_sequence
        )
        self.decoder = ForecastNetDecoder(
            d_model=d_model, nhead_dec=n_head_dec,
            num_layers_dec=num_layers_dec, query_len=query_len, in_dim=in_dim
        )

        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )

    def forward(self, val_ts, vol_ts, mval_ts, beta, pe_ts):
        memory = self.encoder(val_ts, vol_ts, mval_ts, beta, pe_ts)
        decoded = self.decoder(memory, beta, pe_ts)  # pass pe_ts for pooling
        preds = self.pred_head(decoded)
        return preds
