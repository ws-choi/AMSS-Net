import torch
import torch.nn.functional as f


def Pocm_naive(x, gammas, betas):
    """
    :param x: an output feature map of a CNN layer [*, ch, T, F]
    :param gamma: [*, ch, ch]
    :param beta: [*, ch]
    :return: gamma * x + beta
    """
    x = x.unsqueeze(-4)
    gammas = gammas.unsqueeze(-1).unsqueeze(-1)

    pocm = [f.conv2d(x_, gamma_, beta_) for x_, gamma_, beta_ in zip(x, gammas, betas)]

    return torch.cat(pocm, dim=0)


def Pocm_Matmul(x, gammas, betas):
    """
    :param x: an output feature map of a CNN layer [*, ch, T, F]
    :param gamma: [*, ch, ch]
    :param beta: [*, ch]
    :return: gamma * x + beta
    """
    x = x.transpose(-1, -3)  # [*, F, T, ch]
    gammas = gammas.unsqueeze(-3)  # [*, 1, ch, ch]

    pocm = torch.matmul(x, gammas) + betas.unsqueeze(-2).unsqueeze(-3)

    return pocm.transpose(-1, -3)


def Pocm_Matmul_Auto(x, pocm_weight, to_ch, from_ch):
    gammas = pocm_weight[:, :-to_ch].view(-1, from_ch, to_ch)
    betas = pocm_weight[:, -to_ch:]

    return Pocm_Matmul(x, gammas, betas)


def Multi_Head_Pocm(x, pocm_weight, num_head, num_lach):
    """
    :param x: an output feature map of a CNN layer [*, h, T, F, num_lach]
    :param pocm_weight: [*, num_head*(num_lach**2 + num_lach)]
    """

    pocm_weight = pocm_weight.view(-1, num_lach, num_head ** 2 + num_head)
    gammas = pocm_weight[..., :-num_head].view(-1, num_lach, num_head, num_head)  # [*, num_lach, h, h]
    betas = pocm_weight[..., -num_head:]  # [*, num_lach, num_head]

    gammas = gammas.unsqueeze(-3)  # [*, h, 1, num_lach, num_lach]
    betas = betas.unsqueeze(-2).unsqueeze(-2)  # [*, num_head, 1, 1, num_lach]
    pocm = torch.matmul(x.transpose(-1, -4), gammas) + betas

    return pocm.transpose(-1, -4)
