import math
import torch as T
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torchrl.modules import TanhNormal


def logits_to_normal(logits: T.Tensor, beta: float = 1.0) -> Normal:
    loc, scale = T.split(logits, logits.shape[-1] // 2, dim=-1)
    scale = F.softplus(scale, beta=beta)
    return Normal(loc=loc, scale=scale)

def logits_to_tanh_normal(logits: T.Tensor, beta: float = 1.0) -> TanhNormal:
    loc, scale = T.split(logits, logits.shape[-1] // 2, dim=-1)
    scale = F.softplus(scale, beta=beta)
    return TanhNormal(loc=loc, scale=scale)

def jacobian_entropy(dist: Normal):
    log_normalized = 0.5 * math.log(2 * math.pi) + T.log(dist.scale)
    entropy = 0.5 + log_normalized
    entropy = entropy * T.ones_like(dist.loc)
    sample = dist.rsample()
    # entropy = dist.entropy()
    jacobian = 2 * (math.log(2) - sample - F.softplus(-2 * sample))

    return (entropy + jacobian).sum(-1)

def jacobian_log_prob(dist: Normal, sample: T.Tensor):
    log_unnormalized = -0.5 * ((sample - dist.loc) / dist.scale).square()
    log_normalized = 0.5 * math.log(2 * math.pi) + T.log(dist.scale)
    # log_p = dist.log_prob(sample)
    jacobian = 2 * (math.log(2) - sample - F.softplus(-2 * sample))
    # return (log_p - jacobian).sum(dim=-1).mean(dim=0)
    return (log_unnormalized - log_normalized - jacobian).sum(dim=-1).mean()
