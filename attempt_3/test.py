import torch

from volatility_net import VolatilityNet


def main():
    batch_size = 2
    seq_len = 30

    inputs = [torch.randn(batch_size, 1, seq_len, 1) for _ in range(14)]

    model = VolatilityNet(d_model=128)
    mean_logit, std_logit = model(*inputs)

    print("Mean logit shape:", mean_logit.shape)
    print("Std logit shape:", std_logit.shape)
    print("Mean logits:", mean_logit)
    print("Std logits:", std_logit)


if __name__ == "__main__":
    main()
