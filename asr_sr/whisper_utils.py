import torch
import torch.nn.functional as F


def compute_avg_logprob(
    generated_ids: torch.Tensor,
    scores: tuple
) -> list[float]:
    """
    Вычисляет среднюю log-вероятность для сгенерированных последовательностей.

    Args:
        generated_ids: Тензор с id сгенерированных токенов [batch_size, seq_len]
        scores: Кортеж с logits для каждого шага генерации

    Returns:
        Список средних log-вероятностей для каждой последовательности в batch
    """
    prompt_len = 4
    eos_id = 50257

    batch_sz = generated_ids.size(0)
    sum_lp = torch.zeros(batch_sz, device=generated_ids.device)
    tok_count = torch.zeros(batch_sz, device=generated_ids.device)
    active = torch.ones(batch_sz, device=generated_ids.device, dtype=torch.bool)

    # Пропускаем первые prompt_len токенов (спец символы)
    for t, step in enumerate(scores[prompt_len:], start=prompt_len):
        tok = generated_ids[:, t]
        logp = F.log_softmax(step, dim=-1)
        lp = logp.gather(1, tok[:, None]).squeeze(1)

        is_special = tok >= eos_id

        # Count current token while still active (include EOS token itself),
        # then deactivate sequence for subsequent steps.
        add_mask = active & (~is_special)
        sum_lp += lp * add_mask
        tok_count += add_mask.to(tok_count.dtype)
        active = active & (tok != eos_id)

    return (sum_lp / tok_count.clamp(min=1)).cpu().tolist()
