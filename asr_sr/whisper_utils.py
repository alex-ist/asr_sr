import torch
import torch.nn.functional as F


def compute_avg_logprob(
    generated_ids: torch.Tensor,
    scores: tuple
) -> list[float]:
    """
    Compute average log-probability for generated sequences.

    Args:
        generated_ids: Tensor with generated token IDs [batch_size, seq_len]
        scores: Tuple with logits for each generation step

    Returns:
        List of average log-probabilities for each sequence in batch
    """
    prompt_len = 4
    eos_id = 50257

    batch_sz = generated_ids.size(0)
    sum_lp = torch.zeros(batch_sz, device=generated_ids.device)
    tok_count = torch.zeros(batch_sz, device=generated_ids.device)
    active = torch.ones(batch_sz, device=generated_ids.device, dtype=torch.bool)

    # Skip first prompt_len tokens (special symbols)
    for t, step in enumerate(scores[prompt_len:], start=prompt_len+1):
        tok = generated_ids[:, t]
        logp = F.log_softmax(step, dim=-1)
        lp = logp.gather(1, tok[:, None]).squeeze(1)

        # Count current token while still active (include EOS token itself)
        sum_lp += lp * active
        tok_count += active.to(tok_count.dtype)

        # Then deactivate sequence for subsequent steps
        active = active & (tok != eos_id)

    return (sum_lp / tok_count.clamp(min=1)).cpu().tolist()
