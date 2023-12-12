import torch


class CTC(torch.autograd.Function):

    """Docstring for CTC."""

    @staticmethod
    def forward(
        ctx,
        log_probs: torch.tensor,
        input_lengths: torch.tensor,
        trans: torch.tensor,
        trans_lengths: torch.tensor,  # trans_length but not target_lengths
        transform: torch.tensor,
        reduction="sum",
    ):
        B, T, C = log_probs.shape
        S = trans.shape[-1]
        min_value = -1e10

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
                torch.logsumexp(trans + log_alpha[:, t - 1, :].unsqueeze(1), dim=-1)
                + log_probs_token[:, t - 1, :]
            )
            log_beta[:, t, :] = (
                torch.logsumexp(
                    trans.permute(0, 2, 1) + log_beta[:, t - 1, :].unsqueeze(1), dim=-1
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
        for_bak_numerator_new = torch.logsumexp(log_gamma, dim=-1)

        for_bak_loss_new = (
            torch.sum(mask * (for_bak_numerator_new), dim=-1) / input_lengths
        )

        print(
            "for_bak_numerator_new[B_range, input_lengths[B_range]-1]",
            for_bak_numerator_new[B_range, input_lengths[B_range] - 1],
        )
        print(
            "for_bak_numerator_new[B_range, input_lengths[B_range]-2]",
            for_bak_numerator_new[B_range, input_lengths[B_range] - 2],
        )

        print("for_bak_numerator_new", for_bak_numerator_new)
        print("for_bak_loss_new", for_bak_loss_new)
        print("torch.sum(mask, dim=-1)", torch.sum(mask, dim=-1))
        print("input_lengths", input_lengths)

        ctx.save_for_backward(
            log_probs, log_gamma, mask, for_bak_numerator_new, transform
        )

        if reduction == "none":
            return -for_bak_loss_new
        elif reduction == "sum":
            return -torch.sum(for_bak_loss_new)
        elif reduction == "mean":
            return -torch.mean(for_bak_loss_new / trans_lengths)

    @staticmethod
    def backward(ctx, grad_output):
        log_probs, log_gamma, mask, num, transform = ctx.saved_tensors

        print("log_gamma", log_gamma)
        print("torch.exp(log_gamma)", torch.exp(log_gamma))
        print("num", num)

        print("transform", transform)
        print("torch.exp(transform)", torch.exp(transform))

        B, S, C = transform.shape
        _, T, _ = log_gamma.shape

        grad_num_token = -num.unsqueeze(-1) + log_gamma

        print("grad_num_token", grad_num_token)
        print("torch.exp(grad_num_token)[0, 0, :]", torch.exp(grad_num_token)[0, 0, :])
        print("torch.exp(grad_num_token)[0, 1, :]", torch.exp(grad_num_token)[0, 1, :])
        print("torch.exp(grad_num_token)[0, 5, :]", torch.exp(grad_num_token)[0, 5, :])

        expanded_prob_transform = (
            torch.exp(transform)
            .unsqueeze(1)
            .expand(B, T, S, C)
            .permute(0, 1, 3, 2)
            .reshape(B * T, C, S)
        )
        prob_grad_num_token = (
            torch.exp(grad_num_token).unsqueeze(-1).reshape(B * T, S, 1)
        )

        print("expanded_prob_transform[0, 0, :]", expanded_prob_transform[0, 0, :])
        print("expanded_prob_transform[0, 3, :]", expanded_prob_transform[0, 3, :])

        print("prob_grad_num_token[0, :, 0]", prob_grad_num_token[0, :, 0])
        print("prob_grad_num_token[20, :, 0]", prob_grad_num_token[20, :, 0])

        result = torch.matmul(expanded_prob_transform, prob_grad_num_token).reshape(
            B, T, C
        )
        print("result[0, 0, 18]", result[0, 0, 18])
        print("result[0, 20, 18]", result[0, 20, 18])

        grad_num = torch.logsumexp(
            transform.unsqueeze(1).expand(B, T, S, C).permute(0, 1, 3, 2)
            + grad_num_token.unsqueeze(2),
            dim=-1,
        )
        result2 = (
            torch.exp(log_probs + log_probs)
            - torch.exp(log_probs + grad_num) / torch.exp(log_probs)
            - torch.exp(log_probs + log_probs)
        )
        print("result2[0, 0, :]", result2[0, 0, :])
        print("result2[0, 10, :]", result2[0, 10, :])
        print("result2[0, 20, :]", result2[0, 20, :])
        print("torch.sum(result2[0, 0, :])", torch.sum(result2[0, 0, :]))
        print("torch.sum(result2[0, 10, :])", torch.sum(result2[0, 10, :]))
        print("torch.sum(result2[0, 20, :])", torch.sum(result2[0, 20, :]))
        grad_num_masked = mask.unsqueeze(-1) * torch.exp(grad_num)

        grad_input = grad_output * (-grad_num_masked)

        return grad_input, None, None, None, None, None, None


def ctc_loss(
    log_probs: torch.tensor,
    input_lengths: torch.tensor,
    trans: torch.tensor,
    trans_lengths: torch.tensor,
    transform: torch.tensor,
    reduction="sum",
):
    return CTC.apply(
        log_probs, input_lengths, trans, trans_lengths, transform, reduction
    )


def ctc_loss_autograd(
    log_probs: torch.tensor,
    input_lengths: torch.tensor,
    trans: torch.tensor,
    trans_lengths: torch.tensor,  # trans_length but not target_lengths
    transform: torch.tensor,
    reduction="sum",
):
    B, T, C = log_probs.shape
    S = trans.shape[-1]
    min_value = torch.finfo(log_probs.dtype).min

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
            torch.logsumexp(trans + log_alpha[:, t - 1, :].unsqueeze(1), dim=-1)
            + log_probs_token[:, t - 1, :]
        )
        log_beta[:, t, :] = (
            torch.logsumexp(
                trans.permute(0, 2, 1) + log_beta[:, t - 1, :].unsqueeze(1), dim=-1
            )
            + log_probs_token[B_range, input_lengths[B_range] - t, :]
        )

    return -torch.logsumexp(
        torch.stack(
            [
                log_alpha[B_range, input_lengths[B_range], trans_lengths[B_range] - 1],
                log_alpha[B_range, input_lengths[B_range], trans_lengths[B_range] - 2],
            ],
            dim=-1,
        ),
        dim=-1,
    )

    input_lengths_expanded = input_lengths.unsqueeze(-1).unsqueeze(-1).expand(B, T, S)
    time_expanded = (
        torch.arange(T, device=log_probs.device)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(B, T, S)
    )

    indices = input_lengths_expanded - time_expanded - 1
    pos_indices = torch.where(indices >= 0, indices, indices + T).to(dtype=torch.int64)
    log_beta_gathered = torch.gather(log_beta[:, 1:, :], 1, pos_indices)
    log_gamma = log_alpha[:, 1:, :] + log_beta_gathered - log_probs_token
    for_bak_numerator_new = torch.logsumexp(log_gamma, dim=-1)

    for_bak_loss_new = torch.sum(mask * (for_bak_numerator_new), dim=-1) / input_lengths

    print(
        "for_bak_numerator_new[B_range, input_lengths[B_range]-1]",
        for_bak_numerator_new[B_range, input_lengths[B_range] - 1],
    )
    print(
        "for_bak_numerator_new[B_range, input_lengths[B_range]-2]",
        for_bak_numerator_new[B_range, input_lengths[B_range] - 2],
    )

    print("for_bak_numerator_new", for_bak_numerator_new)
    print("for_bak_loss_new", for_bak_loss_new)
    print("torch.sum(mask, dim=-1)", torch.sum(mask, dim=-1))
    print("input_lengths", input_lengths)

    if reduction == "none":
        return -for_bak_loss_new
    elif reduction == "sum":
        return -torch.sum(for_bak_loss_new)
    elif reduction == "mean":
        return -torch.mean(for_bak_loss_new / trans_lengths)
