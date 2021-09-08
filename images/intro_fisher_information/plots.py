import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use('seaborn-white')

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 18,
    "font.size": 18,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    'text.latex.preamble': r'\usepackage{amsmath, amssymb}',
}

plt.rcParams.update(tex_fonts)
plt.rcParams.update({"legend.handlelength": 1.5})
plt.rcParams.update({
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.fancybox": False,
    })


def gaussian(x, mean, std):
    norm = std * math.sqrt(2 * math.pi)
    return (1 / norm) * np.exp(-0.5 * ((x - mean) / std)**2)


def log_gaussian(x, mean, std):
    log_norm = math.log(std * math.sqrt(2 * math.pi))
    return -(log_norm + 0.5 * ((x - mean) / std)**2)


def d_log_gaussian(x, mean, std):
    return (x - mean) / std


def plot_gaussians():
    x = np.arange(-2, 2, 0.01)
    y1 = gaussian(x, 0, 0.25)
    y2 = gaussian(x, 0, 0.5)
    y3 = gaussian(x, 0, 1)
    plt.figure(figsize=(10, 4))
    plt.plot(x, y1, label="$\sigma = 0.25$", linestyle=":", color="k")
    plt.plot(x, y2, label="$\sigma = 0.5$", linestyle="--", color="k")
    plt.plot(x, y3, label="$\sigma = 1.0$", linestyle="-", color="k")
    plt.xlabel("$x$")
    plt.ylabel("$p(x \mid \mu, \sigma)$")
    plt.legend()
    plt.savefig("gaussians.svg", format="svg", bbox_inches="tight")


def plot_fisher_gaussian_steps():
    x = np.arange(-2, 2, 0.01)
    fig, axs = plt.subplots(1, 4, figsize=(22, 4))
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=22)
    axs[0].plot(x, gaussian(x, 0, 1), color='k')
    axs[0].set_title("{\\bf (a)} $p(x \mid \mu, \sigma)$", pad=25, fontsize=30)
    axs[1].plot(x, log_gaussian(x, 0, 1), color='k')
    axs[1].set_title("{\\bf (b)} $\log p(x \mid \mu, \sigma)$", pad=25, fontsize=30)
    axs[2].plot(x, d_log_gaussian(x, 0, 1), color='k')
    axs[2].set_title(
        "{\\bf (c)} $\\frac{d}{d \mu} \log p(x \mid \mu, \sigma)$", pad=25, fontsize=30)
    axs[3].plot(x, d_log_gaussian(x, 0, 1)**2, color='k')
    axs[3].set_title(
        "{\\bf (d)} $\\left( \\frac{d}{d \mu} \log p(x \mid \mu, \sigma) \\right)^2$",
        pad=25, fontsize=30)
    for ax in axs:
        ax.set_xlabel("$x$", fontsize=30)
    plt.savefig(f"fisher_gaussian_steps.svg", format="svg", bbox_inches="tight")


def plot_fisher_gaussian_derivs():
    xs = np.arange(-4, 4, 0.01)
    tangent = np.arange(-0.8, 0.8, 0.01)
    fig, axs = plt.subplots(1, 2, figsize=(16, 4))
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=22)

    axs[0].set_ylim(-5, -0.5)
    axs[1].plot(xs, d_log_gaussian(xs, 0, 1), color='k')
    for x in [-1, 0, 2]:
        axs[0].plot(xs, log_gaussian(xs, x, 1), color='k')
        y = log_gaussian(0, x, 1)
        axs[0].plot(tangent, y + d_log_gaussian(x, 0, 1) * tangent, 'k--', linewidth="1.4")
        axs[0].plot(0, y, 'o', color='k')
        axs[1].plot(x, d_log_gaussian(x, 0, 1), 'o', color='k')

    axs[0].text(-4.2, -2.8, "$x\!=\!-1$", fontsize=20)
    axs[0].text(-2, -3.6, "$x\!=\!0$", fontsize=20)
    axs[0].text(-0.3, -4.4, "$x\!=\!2$", fontsize=20)
    axs[1].text(-0.6, -1.2, "$x\!=\!-1$", fontsize=20)
    axs[1].text(0.4, -0.2, "$x\!=\!0$", fontsize=20)
    axs[1].text(2.4, 1.8, "$x\!=\!2$", fontsize=20)

    axs[0].set_title("{\\bf (a)} $\log p(x \mid \mu, \sigma)$", pad=20, fontsize=26)
    axs[0].set_xlabel("$\mu$", fontsize=26)
    axs[1].set_title(
        "{\\bf (b)} $\\frac{d}{d \mu} \log p(x \mid \mu, \sigma)$", pad=20, fontsize=26)
    axs[1].set_xlabel("$x$", fontsize=26)

    plt.savefig(f"fisher_gaussian_derivs.svg", format="svg", bbox_inches="tight")


def plot_fisher_bernoulli():
    eps = 1e-3
    theta = np.arange(eps, 1, eps)

    fig, axs = plt.subplots(1, 4, figsize=(22, 4))
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.set_xlabel("$\\theta$", fontsize=30)

    # p(x=1|θ) and p(x=0|θ)
    axs[0].plot(theta, theta, 'k-');
    axs[0].plot(theta, 1 - theta, 'k--');
    axs[0].text(-0.03, 0.70, "$x=0$", fontsize=24);
    axs[0].text(0.8, 0.70, "$x=1$", fontsize=24);
    axs[0].set_title("{\\bf (a)} $p(x \mid \\theta)$", pad=25, fontsize=30)

    # log p(x=1|θ) and log p(x=0|θ)
    axs[1].plot(theta, np.log(theta), 'k-');
    axs[1].plot(theta, np.log(1 - theta), 'k--');
    axs[1].set_ylim(-4, 0.1)
    axs[1].text(0.05, -1.2, "$x=0$", fontsize=24);
    axs[1].text(0.75, -1.2, "$x=1$", fontsize=24);
    axs[1].set_title("{\\bf (b)} $\ell (\\theta \mid x)$", pad=25, fontsize=30)

    # d/dθ log p(x=1|θ) and d/dθ log p(x=0|θ)
    axs[2].plot(theta, (1 / theta), 'k-');
    axs[2].plot(theta, (1 / (theta - 1)), 'k--');
    axs[2].set_ylim(-10, 10)
    axs[2].text(0.05, -3.2, "$x=0$", fontsize=24);
    axs[2].text(0.8, 2.1, "$x=1$", fontsize=24);
    axs[2].set_title(
        "{\\bf (c)} $\ell^\prime (\\theta \mid x)$", pad=25, fontsize=30)

    # E[(d/dθ log p(x|θ))^2]
    axs[3].plot(theta, 1 / (theta* (1 - theta)), 'k-');
    axs[3].set_ylim(3, 20)
    axs[3].set_title(
        "{\\bf (d)} $\\mathbb{E}\left[\ell^\prime(\\theta \mid x)^2\\right]$",
        pad=25, fontsize=30)

    plt.savefig(f"fisher_bernoulli.svg", format="svg", bbox_inches="tight")


if __name__ == "__main__":
    plot_gaussians()
    plot_fisher_gaussian_steps()
    plot_fisher_gaussian_derivs()
    plot_fisher_bernoulli()
