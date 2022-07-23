import torch


def compute_align(log_probs: torch.tensor,
                  input_lengths: torch.tensor,
                  trans: torch.tensor,
                  trans_lengths: torch.tensor,
                  transform: torch.tensor):

    B, T, C = log_probs.shape
    S = trans.shape[-1]
    min_value = torch.finfo(log_probs.dtype).min / 100

    input_lengths = input_lengths.to(device=log_probs.device)
    log_probs_token = torch.logsumexp(transform.unsqueeze(
        1).expand(B, T, S, C) + log_probs.unsqueeze(2), dim=-1)

    B_range = torch.arange(0, B, dtype=torch.long, device=log_probs.device)


    # Initialise alpha
    log_alpha = torch.full((B, T+1, S),
                           min_value,
                           dtype=log_probs.dtype,
                           device=log_probs.device)
    log_alpha[:, 0, 0] = 0.
    log_beta = torch.full_like(log_alpha,
                               min_value,
                               dtype=log_probs.dtype,
                               device=log_probs.device)
    log_beta[B_range, 0, trans_lengths[B_range]-1] = 0.

    for t in range(1, T+1):
        log_alpha[:, t, :] = torch.logsumexp(
            trans + log_alpha[:, t-1, :].unsqueeze(1), dim=-1) + log_probs_token[:, t-1, :]
        log_beta[:, t, :] = torch.logsumexp(trans.permute(
            0, 2, 1) + log_beta[:, t-1, :].unsqueeze(1), dim=-1) + log_probs_token[B_range, input_lengths[B_range]-t, :]

    input_lengths_expanded = input_lengths.unsqueeze(
        -1).unsqueeze(-1).expand(B, T, S)
    time_expanded = torch.arange(T, device=log_probs.device).unsqueeze(
        0).unsqueeze(-1).expand(B, T, S)

    indices = input_lengths_expanded - time_expanded - 1
    pos_indices = torch.where(
        indices >= 0, indices, indices + T).to(dtype=torch.int64)
    log_beta_gathered = torch.gather(
        log_beta[:, 1:, :], 1, pos_indices)
    log_gamma = log_alpha[:, 1:, :] + log_beta_gathered - log_probs_token

    for_bak_numerator = torch.logsumexp(log_gamma, dim=-1)

    log_gamma_norm = log_gamma - for_bak_numerator.unsqueeze(-1)

    return log_gamma_norm


def compute_soft_prior():
    pass


def compute_count_prior():
    pass
