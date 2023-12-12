import torch


class CTC_Softmax(torch.autograd.Function):

    """Docstring for CTC_Softmax."""

    @staticmethod
    def forward(
        ctx,
        log_probs: torch.tensor,
        input_lengths: torch.tensor,
        trans: torch.tensor,
        trans_lengths: torch.tensor,
        transform: torch.tensor,
        eta=1.0,
        reduction="sum",
    ):
        B, T, C = log_probs.shape
        S = trans.shape[-1]

        min_value = torch.finfo(log_probs.dtype).min / 100
        eta = torch.tensor(eta)

        input_lengths = input_lengths.to(device=log_probs.device)
        log_probs_token = torch.logsumexp(
            transform.unsqueeze(1).expand(B, T, S, C) + log_probs.unsqueeze(2), dim=-1
        )

        B_range = torch.arange(0, B, dtype=torch.long, device=log_probs.device)

        mask = torch.arange(T, device=log_probs.device).expand(
            B, -1
        ) < input_lengths.unsqueeze(1)

        # Initialise alpha
        log_alpha = torch.full(
            (B, T + 1, S), min_value, dtype=log_probs.dtype, device=log_probs.device
        )
        log_alpha[:, 0, 0] = 0.0
        log_beta = torch.full_like(
            log_alpha, min_value, dtype=log_probs.dtype, device=log_probs.device
        )
        log_beta[B_range, 0, trans_lengths[B_range] - 1] = 0.0

        for t in range(1, T + 1):
            log_alpha[:, t, :] = (
                1
                / eta
                * torch.logsumexp(
                    eta * (trans + log_alpha[:, t - 1, :].unsqueeze(1)), dim=-1
                )
                + log_probs_token[:, t - 1, :]
            )
            log_beta[:, t, :] = (
                1
                / eta
                * torch.logsumexp(
                    eta * (trans.permute(0, 2, 1) + log_beta[:, t - 1, :].unsqueeze(1)),
                    dim=-1,
                )
                + log_probs_token[B_range, input_lengths[B_range] - t, :]
            )

        input_lengths_expanded = (
            input_lengths.unsqueeze(-1).unsqueeze(-1).expand(B, T, S)
        )
        time_expanded = (
            torch.arange(T, device=log_probs.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(B, T, S)
        )

        indices = input_lengths_expanded - time_expanded - 1
        pos_indices = torch.where(indices >= 0, indices, indices + T).to(
            dtype=torch.int64
        )
        log_beta_gathered = torch.gather(log_beta[:, 1:, :], 1, pos_indices)
        log_gamma = log_alpha[:, 1:, :] + log_beta_gathered - log_probs_token
        for_bak_numerator = 1 / eta * torch.logsumexp(eta * (log_gamma), dim=-1)

        for_bak_loss = torch.sum(mask * (for_bak_numerator), dim=-1) / input_lengths

        ctx.save_for_backward(log_gamma, mask, for_bak_numerator, transform, eta)

        if reduction == "none":
            return -for_bak_loss
        elif reduction == "sum":
            return -torch.sum(for_bak_loss)
        elif reduction == "mean":
            return -torch.mean(for_bak_loss / trans_lengths)

    @staticmethod
    def backward(ctx, grad_output):
        log_gamma, mask, num, transform, eta = ctx.saved_tensors

        B, S, C = transform.shape
        _, T, _ = log_gamma.shape

        grad_num_token = eta * (-num.unsqueeze(-1) + log_gamma)
        grad_num = torch.logsumexp(
            transform.unsqueeze(1).expand(B, T, S, C).permute(0, 1, 3, 2)
            + grad_num_token.unsqueeze(2),
            dim=-1,
        )
        grad_num_masked = mask.unsqueeze(-1) * torch.exp(grad_num)

        grad_input = grad_output * (-grad_num_masked)

        return grad_input, None, None, None, None, None, None


def ctc_softmax(
    log_probs: torch.tensor,
    input_lengths: torch.tensor,
    trans: torch.tensor,
    transform: torch.tensor,
    trans_lengths: torch.tensor,
    eta=1.0,
    reduction="sum",
):
    return CTC_Softmax.apply(
        log_probs, input_lengths, trans, trans_lengths, transform, eta, reduction
    )
