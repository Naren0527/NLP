"""
Unit tests for src/model_engine.py.
Runs offline using mocks without loading Gemma or requiring CUDA.
"""

from unittest.mock import MagicMock, patch
import pytest
import torch

from src.model_engine import ModelEngine, GenerationOutput


class TestModelEngineConfigAndInit:
    """Tests configuration loading and singleton patterns without GPU."""

    def test_config_loading(self):
        engine = ModelEngine(config_path="configs/model.yaml")
        assert engine.model_id == "google/gemma-3-4b-it"
        assert engine.config["generation"]["do_sample"] is False
        assert engine.config["model"]["quantization"]["load_in_4bit"] is True

    def test_singleton_accessor(self):
        engine1 = ModelEngine.get_instance(config_path="configs/model.yaml")
        engine2 = ModelEngine.get_instance(config_path="configs/model.yaml")
        assert engine1 is engine2

    def test_get_gpu_memory_status_cpu_fallback(self):
        engine = ModelEngine()
        status = engine.get_gpu_memory_status()
        assert "allocated_mb" in status
        assert "reserved_mb" in status
        assert "peak_mb" in status


class TestModelEngineGenerationMocked:
    """Verifies generation logic, prompt exclusion, and logit alignment via mocked tensors."""

    @patch("torch.cuda.is_available", return_value=True)
    def test_generate_logic_with_mock_model(self, mock_cuda):
        engine = ModelEngine()
        engine.device = "cuda"
        engine._is_loaded = True

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<user> What is 2+2? <model>"
        
        # Tokenizer call returns batch with 4 prompt tokens: [100, 101, 102, 103]
        mock_input_ids = torch.tensor([[100, 101, 102, 103]], device="cpu")
        mock_tokenizer.return_value = MagicMock(input_ids=mock_input_ids, to=lambda d: MagicMock(input_ids=mock_input_ids))
        
        # Generated sequence has 4 prompt tokens + 3 generated tokens [201, 202, 203] = length 7
        mock_sequence = torch.tensor([100, 101, 102, 103, 201, 202, 203])

        # Step logits: 3 steps, vocab size 500
        step_logits_1 = torch.zeros(1, 500)
        step_logits_1[0, 201] = 5.0  # high logit for token 201
        step_logits_2 = torch.zeros(1, 500)
        step_logits_2[0, 202] = 4.0
        step_logits_3 = torch.zeros(1, 500)
        step_logits_3[0, 203] = 6.0

        mock_logits = (step_logits_1, step_logits_2, step_logits_3)

        mock_output = MagicMock()
        mock_output.sequences = [mock_sequence]
        mock_output.logits = mock_logits
        mock_output.scores = mock_logits

        mock_model = MagicMock()
        mock_model.generate.return_value = mock_output
        mock_model.config._commit_hash = "mock_commit_123"

        mock_tokenizer.decode.side_effect = lambda ids, **kwargs: "4." if 201 in ids else f"tok_{ids[0]}"

        engine.tokenizer = mock_tokenizer
        engine.model = mock_model
        engine.model_revision = "mock_commit_123"

        output = engine.generate("What is 2+2?")

        assert isinstance(output, GenerationOutput)
        assert output.prompt_token_count == 4
        assert output.generated_token_count == 3
        assert output.total_token_count == 7
        assert output.generated_token_ids == [201, 202, 203]
        assert len(output.token_scores) == 3
        assert output.logits_type == "unprocessed_logits (RAW)"
        assert output.mean_token_probability > 0.0
        assert output.geometric_probability > 0.0
