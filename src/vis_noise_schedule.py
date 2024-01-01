import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import load_config_from_yaml
from ddim import DDIM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=Path)
    args = parser.parse_args()

    config = load_config_from_yaml(args.config_path)
    ddim = DDIM.from_config(config)

    alpha = ddim.alpha
    sigma = torch.sqrt(1 - alpha**2)
    log_snr = 2 * (torch.log10(alpha) - torch.log10(sigma))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4), dpi=200)
    fig.suptitle(f"Noise Schedule for {str(args.config_path)}")

    ax1.plot(np.arange(1, config.t + 1), alpha[1:].cpu().numpy())
    ax1.set_xlabel("t")
    ax1.set_ylabel("alpha")

    ax2.plot(np.arange(1, config.t + 1), sigma[1:].cpu().numpy())
    ax2.set_xlabel("t")
    ax2.set_ylabel("sigma")

    ax3.plot(np.arange(1, config.t + 1), log_snr[1:].cpu().numpy())
    ax3.set_xlabel("t")
    ax3.set_ylabel("log(SNR)")

    output_path = Path("./vis/noise_schedule.png")
    print(f"Saving to {output_path}")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    main()
