import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
import clip
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F


class SparseFCN(nn.Module):
    def __init__(self, input_dim, output_dim, sparsity=0.9):
        super(SparseFCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity = sparsity

        # 定义稀疏权重矩阵，并随机初始化
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)
        self.mask = nn.Parameter(torch.zeros_like(self.weight))

        # 根据sparsity参数生成稀疏掩码
        self._create_mask()

    def _create_mask(self):
        with torch.no_grad():
            num_zeros = int(self.weight.numel() * (1 - self.sparsity))
            flat_weight = self.weight.abs().flatten()
            _, indices = torch.topk(flat_weight, k=num_zeros)
            self.mask.data.flatten()[indices] = 1

    def forward(self, x):
        # 对权重进行稀疏化处理
        sparse_weight = self.weight * self.mask

        # 使用矩阵乘法实现前向传播
        out = torch.matmul(x, sparse_weight)

        return out


class SparseLinearV2(nn.Module):
    def __init__(self, in_features, out_features, sparsity=0.5):
        super(SparseLinearV2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.mask = nn.Parameter(torch.ones(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.uniform_(self.mask, 0, 1)
        self.mask.data = nn.functional.threshold(self.mask.data, self.sparsity, 0)

    def forward(self, input):
        masked_weight = self.weight * self.mask
        output = nn.functional.linear(input, masked_weight, self.bias)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, sparsity={}'.format(
            self.in_features, self.out_features, self.sparsity)


class GaussianLinear(nn.Module):
    def __init__(self, in_features, out_features, mean_init=0.0, std_init=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mean = nn.Parameter(torch.Tensor(out_features, in_features).fill_(mean_init))
        self.log_stddev = nn.Parameter(torch.Tensor(out_features, in_features).fill_(math.log(std_init)))

    def forward(self, x):
        weight = torch.distributions.Normal(self.mean, self.log_stddev.exp()).sample()
        return F.linear(x, weight)


class FastGaussianLinear(nn.Module):
    def __init__(self, in_features, out_features, mean_init=0.0, std_init=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mean_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(mean_init))
        self.log_stddev_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(math.log(std_init)))

        self.register_buffer("eps", torch.randn(out_features))

    def forward(self, x):
        mean = F.linear(x, self.mean_weight)
        stddev = torch.exp(F.linear(x, self.log_stddev_weight))
        weight = mean + stddev * self.eps.view(-1, 1)
        return F.linear(x, weight)


class LaplacianLinear(nn.Module):
    def __init__(self, in_features, out_features, loc_init=0.0, scale_init=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.loc = nn.Parameter(torch.Tensor(out_features, in_features).fill_(loc_init))
        self.log_scale = nn.Parameter(torch.Tensor(out_features, in_features).fill_(math.log(scale_init)))

    def forward(self, x):
        weight = torch.distributions.Laplace(self.loc, self.log_scale.exp()).sample()
        return F.linear(x, weight)


class CauchyLinear(nn.Module):
    def __init__(self, in_features, out_features, loc_init=0.0, scale_init=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.loc = nn.Parameter(torch.Tensor(out_features, in_features).fill_(loc_init))
        self.log_scale = nn.Parameter(torch.Tensor(out_features, in_features).fill_(math.log(scale_init)))

    def forward(self, x):
        weight = torch.distributions.Cauchy(self.loc, self.log_scale.exp()).sample()
        return F.linear(x, weight)


class StudentTLinear(nn.Module):
    def __init__(self, in_features, out_features, df_init=1.0, loc_init=0.0, scale_init=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.df = nn.Parameter(torch.Tensor([df_init]))
        self.loc = nn.Parameter(torch.Tensor(out_features, in_features).fill_(loc_init))
        self.log_scale = nn.Parameter(torch.Tensor(out_features, in_features).fill_(math.log(scale_init)))

    def forward(self, x):
        weight = torch.distributions.StudentT(df=self.df, loc=self.loc, scale=self.log_scale.exp()).sample()
        return F.linear(x, weight)


class MixtureGaussianLinear(nn.Module):
    def __init__(self, in_features, out_features, num_mixtures=5, mean_init=0.0, std_init=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_mixtures = num_mixtures
        self.means = nn.Parameter(torch.Tensor(num_mixtures, out_features, in_features).fill_(mean_init))
        self.log_stddevs = nn.Parameter(torch.Tensor(num_mixtures, out_features, in_features).fill_(math.log(std_init)))
        self.mixture_weights = nn.Parameter(torch.Tensor(num_mixtures).fill_(1 / num_mixtures))

    def forward(self, x):
        weights = torch.zeros_like(self.means)
        for i in range(self.num_mixtures):
            weights[i] = torch.distributions.Normal(self.means[i], self.log_stddevs[i].exp()).sample()
        weight = torch.sum(weights * self.mixture_weights.view(-1, 1, 1), dim=0)
        return F.linear(x, weight)


class GammaLinear(nn.Module):
    def __init__(self, in_features, out_features, concentration_init=1.0, rate_init=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concentration = nn.Parameter(torch.Tensor(out_features, in_features).fill_(concentration_init))
        self.rate = nn.Parameter(torch.Tensor(out_features, in_features).fill_(rate_init))

    def forward(self, x):
        weight = torch.distributions.Gamma(self.concentration, self.rate).sample()
        return F.linear(x, weight)


class BetaLinear(nn.Module):
    def __init__(self, in_features, out_features, alpha_init=1.0, beta_init=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = nn.Parameter(torch.Tensor(out_features, in_features).fill_(alpha_init))
        self.beta = nn.Parameter(torch.Tensor(out_features, in_features).fill_(beta_init))

    def forward(self, x):
        weight = torch.distributions.Beta(self.alpha, self.beta).sample()
        return F.linear(x, weight)


class BayesianLinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, prior_mean=0, prior_std=1):
        super(BayesianLinearLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        # 定义权重的后验分布参数
        self.weight_mu = nn.Parameter(torch.Tensor(output_dim, input_dim).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(output_dim, input_dim).uniform_(-5, -4))
        self.weight_prior = torch.distributions.Normal(prior_mean, prior_std)

    def forward(self, x):
        # 从正态分布中采样权重
        weight_epsilon = self.weight_prior.sample(self.weight_mu.size())
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight = self.weight_mu + weight_sigma * weight_epsilon
        return torch.matmul(x, weight.t())


class ReparameterizationLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True):
        """
        Implements Linear layer with reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_features: int -> size of each input sample,
            out_features: int -> size of each output sample,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super(ReparameterizationLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init,  # mean of weight
        # variance of weight --> sigma = log (1 + exp(rho))
        self.posterior_rho_init = posterior_rho_init,
        self.bias = bias

        self.mu_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.rho_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('eps_weight',
                             torch.Tensor(out_features, in_features),
                             persistent=False)
        self.register_buffer('prior_weight_mu',
                             torch.Tensor(out_features, in_features),
                             persistent=False)
        self.register_buffer('prior_weight_sigma',
                             torch.Tensor(out_features, in_features),
                             persistent=False)
        if bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_features))
            self.rho_bias = nn.Parameter(torch.Tensor(out_features))
            self.register_buffer(
                'eps_bias',
                torch.Tensor(out_features),
                persistent=False)
            self.register_buffer(
                'prior_bias_mu',
                torch.Tensor(out_features),
                persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_features),
                                 persistent=False)
        else:
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)

        self.init_parameters()
        self.quant_prepare = False

    def init_parameters(self):
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        self.mu_weight.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
        self.rho_weight.data.normal_(mean=self.posterior_rho_init[0], std=0.1)
        if self.mu_bias is not None:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)
            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0],
                                       std=0.1)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        kl = self.kl_div(
            self.mu_weight,
            sigma_weight,
            self.prior_weight_mu,
            self.prior_weight_sigma)
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(self.mu_bias, sigma_bias,
                              self.prior_bias_mu, self.prior_bias_sigma)
        return kl

    def forward(self, input, return_kl=False):
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        eps_weight = self.eps_weight.data.normal_()
        tmp_result = sigma_weight * eps_weight
        weight = self.mu_weight + tmp_result

        if return_kl:
            kl_weight = self.kl_div(self.mu_weight, sigma_weight,
                                    self.prior_weight_mu, self.prior_weight_sigma)
        bias = None

        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = self.mu_bias + (sigma_bias * self.eps_bias.data.normal_())
            if return_kl:
                kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        out = F.linear(input, weight, bias)

        if return_kl:
            if self.mu_bias is not None:
                kl = kl_weight + kl_bias
            else:
                kl = kl_weight

            return out, kl

        return out
