import torch

from generative_recommender_impl import GenerativeRecommender


def main() -> None:
    torch.manual_seed(0)

    model = GenerativeRecommender(d_in_enc=512)
    model.eval()

    batch_size = 2
    seq_len = 7
    input_emb = torch.randn(batch_size, seq_len, 512)
    enc_mask_seq = torch.tensor(
        [
            [1] * 7,
            [1] * 7,
        ],
        dtype=torch.long,
    )
    target_sid = torch.tensor([[1, 2, 3], [10, 20, 30]], dtype=torch.long)
    model.set_valid_prefixes(
        [
            torch.tensor([[1], [10], [11]], dtype=torch.long),
            torch.tensor([[1, 2], [10, 20], [11, 12]], dtype=torch.long),
            torch.tensor([[1, 2, 3], [10, 20, 30], [11, 12, 13]], dtype=torch.long),
        ]
    )

    with torch.no_grad():
        out = model(
            input_emb=input_emb,
            enc_mask_seq=enc_mask_seq,
            target_sid=target_sid,
        )

    assert out["ctr_logits"].shape == (batch_size, 3, 256)
    assert out["h_next"].shape == (batch_size, 3, 512)
    assert out["h_full"].shape == (batch_size, seq_len + 3, 512)
    assert out["loss"].ndim == 0
    assert out["ntp_loss"].ndim == 0
    assert out["discriminator_loss"].ndim == 0
    assert len(out["discriminator_loss_by_depth"]) == 3
    assert len(out["discriminator_logits_by_depth"]) == 3
    assert len(out["discriminator_metrics_by_depth"]) == 3

    bos_ids = torch.zeros(batch_size, 1, dtype=torch.long)
    sid0_global = torch.tensor([[18 + 5], [18 + 6]], dtype=torch.long)
    sid1_global = torch.tensor([[274 + 7], [274 + 8]], dtype=torch.long)
    with torch.no_grad():
        step0_out = model._decode_query_ids(input_emb, bos_ids, enc_mask_seq)
        step1_out = model._decode_query_ids(input_emb, torch.cat([bos_ids, sid0_global], dim=1), enc_mask_seq)
        step2_out = model._decode_query_ids(
            input_emb,
            torch.cat([bos_ids, sid0_global, sid1_global], dim=1),
            enc_mask_seq,
        )
    assert step0_out.shape == (batch_size, seq_len + 1, 512)
    assert step1_out.shape == (batch_size, seq_len + 2, 512)
    assert step2_out.shape == (batch_size, seq_len + 3, 512)

    with torch.no_grad():
        greedy_ids, greedy_scores = model.generate(input_emb=input_emb, enc_mask_seq=enc_mask_seq)
        beam_ids, beam_scores = model.generate(
            input_emb=input_emb,
            enc_mask_seq=enc_mask_seq,
            beam_sizes=[2, 3, 4],
            temperature=1.0,
        )

    assert greedy_ids.shape == (batch_size, 3)
    assert greedy_scores.shape == (batch_size,)
    assert beam_ids.shape == (batch_size, 4, 3)
    assert beam_scores.shape == (batch_size, 4)
    assert torch.all((greedy_ids >= 0) & (greedy_ids < 256))
    assert torch.all((beam_ids >= 0) & (beam_ids < 256))

    print("Self check passed.")


if __name__ == "__main__":
    main()
