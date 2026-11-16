"""The Qwen3.8 MHA adapter must use the v0.5.21 prefill API."""

from types import SimpleNamespace

import torch

from atom.plugin.sglang.attention_backend.full_attention.full_attention_backend import (
    ATOMAttnBackendForSgl,
)


def test_extend_uses_v0521_updater_without_plan():
    seen = []

    class Updater:
        kv_indptr = torch.tensor([0, 5])
        kv_indices = torch.arange(5)

        def update(
            self,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            prefix_lens,
            encoder_lens,
            spec_info,
        ):
            seen.append((seq_lens_sum, spec_info))

    backend = SimpleNamespace(
        indices_updater_prefill=Updater(),
        qo_indptr=torch.tensor([0, 3, 99]),
        _max_len=lambda cpu, gpu: max(cpu),
    )
    batch = SimpleNamespace(
        req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([5]),
        seq_lens_sum=5,
        extend_prefix_lens=torch.tensor([2]),
        encoder_lens=None,
        extend_seq_lens_cpu=[3],
        extend_seq_lens=torch.tensor([3]),
        seq_lens_cpu=[5],
    )
    ATOMAttnBackendForSgl._init_extend_mha(backend, 1, batch)
    assert seen == [(5, None)]
    assert (
        backend.forward_metadata.kv_indices
        is backend.indices_updater_prefill.kv_indices
    )
    assert backend.forward_metadata.qo_indptr.tolist() == [0, 3]


def test_draft_extend_passes_request_table_and_keeps_generated_layout():
    req_table = torch.arange(8).view(1, 8)
    indices, kv_indptr, qo_indptr = (
        torch.arange(5),
        torch.tensor([0, 5]),
        torch.tensor([0, 3]),
    )

    class Spec:
        num_tokens_per_req = 3

        def generate_attn_arg_prefill(
            self,
            *,
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            req_to_token,
        ):
            assert req_to_token is req_table
            assert paged_kernel_lens_sum == 5
            return indices, kv_indptr, qo_indptr, None

    backend = SimpleNamespace(
        req_to_token=req_table, _max_len=lambda cpu, gpu: max(cpu)
    )
    batch = SimpleNamespace(
        spec_info=Spec(),
        req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([5]),
        seq_lens_sum=5,
        seq_lens_cpu=[5],
    )
    ATOMAttnBackendForSgl._init_draft_extend_v2_mha(backend, 1, batch)
    assert backend.forward_metadata.kv_indices is indices
    assert backend.forward_metadata.qo_indptr is qo_indptr
    assert backend.forward_metadata.max_extend_len == 3
