'''
network that predicts a volatility score for the 7 days following time t, the ground truth is calculated as follows:

 μ = 1/2 * Σᵢ₌₁⁷ [ (log(S_i / S_{i-1})^2 ] + 1/14 * Σᵢ₌₁⁷ [(log(S_i/S_0))^2]
 
 the first term is daily log return
 second term is cumulative log displacement from the day we are predicting
 
 that mu along with a standard variation of that prediction is going to be what the model predicts so that we can later
 use it in a buy/sell/hold signal calculation for straddle options
 
 the model will take as input (all of which will be time series over past 30 days, i.e. there will be 30 t's from which we will
 calculate these values):
 
 log return: log(C_t/C_{t-1}) shape: (B, 1, 30, 1)
 
 squared return: (log(C_t/C_{t-1}))^2 shape: (B, 1, 30, 1)
 
 hl_range_pct: (Hight_t - Low_t)/close_{t-1} shape: (B, 1, 30, 1)
 
 close open return: ln(C_t/O_t) shape: (B, 1, 30, 1)
 
 parkinson var: 1/(4 ln(2)) (ln(High_t/Low_t))^2 shape: (B, 1, 30, 1)
 
 turnover ***: ln(Volume_t * close_t): shape: (B, 1, 30, 1)
 
 gap return: ln(O_t/C_{t-1}) shape: (B, 1, 30, 1)
 
 garman klass: 1/2 (ln(High_t/Low_t))^2 - (2 ln(2))(ln(C_t/O_t))^2 shape: (B, 1, 30, 1)
 
 past 5 day sum of squared return: Σ [(log(S_{t-i}/S_{t-i-1}))^2] shape: (B, 1, 30, 1)
 
 past 5 day sum of squared displacement: Σ [(log(S_{t-i}/S_{t-6}))^2] shape: (B, 1, 30, 1)
 
 past 10 day sum of squared return, shape: (B, 1, 30, 1)
 
 past 10 day sum of squared displacement, shape: (B, 1, 30, 1)
 
 past 20 day sum of squared return, shape: (B, 1, 30, 1)
 
 past 20 day sum of squared displacement, shape: (B, 1, 30, 1)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import CrossAttentionTransformer, SelfAttentionTransformer

class SwiGLU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


class VolatilityNet(nn.Module):
    def __init__(self, d_model, ):
        super().__init__()
        
        self.T = SelfAttentionTransformer(d_model=d_model, 
                                          depth=6, 
                                          heads=8, 
                                          mlp_ratio=4,
                                          dropout=0.1, 
                                          attn_dropout=0.1, 
                                          num_modalities=14, 
                                          max_time=30, 
                                          value_dim=1, 
                                          input_dim=4, 
                                          use_rope=True)
        self.mean_refiner = CrossAttentionTransformer(
            d_model=d_model,
            depth=4,
            heads=8,
            mlp_ratio=4,
            dropout=0.1,
            attn_dropout=0.1,
            input_dim=d_model,
        )
        self.std_refiner = CrossAttentionTransformer(
            d_model=d_model,
            depth=4,
            heads=8,
            mlp_ratio=4,
            dropout=0.1,
            attn_dropout=0.1,
            input_dim=d_model,
        )
        self.mean_head = nn.Linear(d_model, 1)
        self.std_head = nn.Linear(d_model, 1)
    
    def forward(self, log_return, 
                squared_return, 
                hl_range_pct, 
                close_open_return,
                parkinson_var,
                turnover,
                gap_return,
                garman_klass,
                ret_5d,
                dis_5d,
                ret_10d,
                dis_10d,
                ret_20d,
                dis_20d):
        
        modalities = [
            log_return,
            squared_return,
            hl_range_pct,
            close_open_return,
            parkinson_var,
            turnover,
            gap_return,
            garman_klass,
            ret_5d,
            dis_5d,
            ret_10d,
            dis_10d,
            ret_20d,
            dis_20d,
        ]
        comp = torch.cat(modalities, dim=1)           # [B, 14, T, value_dim]
        comp = comp.permute(0, 2, 1, 3).contiguous()  # [B, T, 14, value_dim]
        encoded_comp = self.T(comp, attn_mask=None, is_causal=False)
        mean_token = self.mean_refiner(encoded_comp)
        std_token = self.std_refiner(encoded_comp)
        mean_logit = self.mean_head(mean_token).squeeze(-1)
        std_logit = self.std_head(std_token).squeeze(-1)
        return mean_logit, std_logit
