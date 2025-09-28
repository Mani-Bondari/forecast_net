"""_summary_
forecast_net is designed to use a variety of information sources to forcast trends in the
stock market over a 7 (trading) day period. while hedgefunds billion dollar firms have access
to financial information at lower latency than the average folk, us average folk can take
advantage of the vast variety of media and information channels at our disposal such as:
- yahoo finance
- stocktwits
- reddit
- instagram comments
- threads
- tik tok comments
- etc

with a selective picky scraping system using a mix of api's, web-scraping, custom ranking
algorithms etc. we can collect a vast variety of information from a wide array of sources
broadening our model in the breadth aspect as opposed to the depth that large firms have.
this heightened breadth should make up for the increased latency of the acquisition of this 
information.


financial info we will be using:
--------------------------------
60 day return history of the stock (close value - open value) represented as a tensor (B, 30, 1)

30 day volume history (volume for each of the previous 30 days ligned up with the above) (B, 30, 1)

30 day market value history (related to the return history, 
but can let us know if there were any stock splits which can impact the forecast) (B, 30, 1)

yahoo finance's calculation of the most recent beta value to express the volatility of the stock (B, 1, 1)

30 day history of the P/E value used to calculate how the stock moves in general (B, 30, 1)

social info we will be using:
-------------------------------
reddit:
we will scrape a list of 5000 top reddit posts from a certain date interval related to the ticker 
symbol in question, we will use the following ranking algorithm to rank each of the posts to filter out noise:

u = upvotes
c = # comments
l = decay rate of decay function e^-l(dt)
k = author karma
r = sentiment relevence score

Score(post) = (a * norm(u) + b * norm(c)) * (1 + l * log(1 + k)) * e^(-l(dt)) * r

a, b, l, r are all learnable parameters by the model, so it can learn over a long period of time and looking 
at data what factors contribute to a higher reliability and relevance score for the task at hand.



    
"""

import torch
import torch.nn as nn

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
    def __init__(self, d_model:int = 1024, max_len:int = 10000):
        self.sinusoidal_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
        self.learnable_bias = nn.Parameter(torch.zeros(1, max_len, d_model))
    
    def forward(self, x: torch.Tensor):
        return self.sinusoidal_encoding(x) + self.learnable_bias[:, :x.size(1)]

class ForecastNetEncoder(nn.Module):
    def __init__(self, d_model:int = 1024, in_dim:int = 1, nhead:int = 8, num_layers=16):
        super().__init__()
        
        self.val_ts_proj = nn.Linear(in_dim, d_model)
        
        self.vol_ts_proj = nn.Linear(in_dim, d_model)
        
        self.mval_ts_proj = nn.Linear(in_dim, d_model)
        
        self.beta_proj = nn.Linear(in_dim, d_model)
        
        self.pe_ts_proj = nn.Linear(in_dim, d_model)
        
        self.lang_channel_shuffler_reddit = nn.Linear(d_model, d_model)
        
        
        forecast_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.forecast_encoder = nn.TransformerEncoder(encoder_layer=forecast_encoder_layer, num_layers=num_layers)
        
    def forward(self, val_ts, vol_ts, mval_ts, beta, pe_ts, lang_vec_reddit):
        
        val_ts_proj = self.val_ts_proj(val_ts)
        vol_ts_proj = self.vol_ts_proj(vol_ts)
        mval_ts_proj = self.mval_ts_proj(mval_ts)
        beta_proj = self.beta_proj(beta)
        pe_ts_proj = self.pe_ts_proj(pe_ts)
        lang_vec_reddit_shuff = self.lang_channel_shuffler_reddit(lang_vec_reddit)
        
        forecast_tens = torch.cat((val_ts_proj, vol_ts_proj, mval_ts_proj, beta_proj, pe_ts_proj, lang_vec_reddit_shuff), dim=1)
        
        encoded_forecast_tens = self.forecast_encoder(forecast_tens)
        
        return encoded_forecast_tens

