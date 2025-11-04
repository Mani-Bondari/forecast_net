import torch

from model_1.forecast_net import ForecastNet
# Import your ForecastNet here if in a separate file:
# from forecast_net import ForecastNet

def test_forecast_net():
    B = 4   # batch size
    d_model = 1024  # reduce for test speed
    query_len = 7

    # Expected input shapes:
    # val_ts   : (B, 30, 1)
    # vol_ts   : (B, 30, 1)
    # mval_ts  : (B, 30, 1)
    # beta     : (B, 1, 1)
    # pe_ts    : (B, 30, 1)

    val_ts = torch.randn(B, 30, 1)
    vol_ts = torch.randn(B, 30, 1)
    mval_ts = torch.randn(B, 30, 1)
    beta = torch.randn(B, 1, 1)
    pe_ts = torch.randn(B, 30, 1)

    # Instantiate model
    model = ForecastNet(
        in_dim=1,
        d_model=d_model,
        n_head_enc=4,
        n_head_dec=4,
        num_layers_enc=16,   # reduce layers for quick test
        num_layers_dec=16,
        query_len=query_len,
        max_len=1000
    )

    # Run forward pass
    preds = model(val_ts, vol_ts, mval_ts, beta, pe_ts)

    print("Input shapes:")
    print(f"val_ts   : {val_ts.shape}")
    print(f"vol_ts   : {vol_ts.shape}")
    print(f"mval_ts  : {mval_ts.shape}")
    print(f"beta     : {beta.shape}")
    print(f"pe_ts    : {pe_ts.shape}")
    print("\nOutput:")
    print(f"preds    : {preds.shape}")  # Expected (B, query_len, 1)


if __name__ == "__main__":
    test_forecast_net()
