# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline regression and compatibility checks for the embedding dependency stack.

Run with: HF_HUB_OFFLINE=1 python -m unittest discover -s tests -v
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ap_embeddings import load_sentence_transformer


class EmbeddingSecurityTest(unittest.TestCase):
    def test_untrusted_local_custom_module_is_not_imported(self):
        # This module only writes a harmless marker inside the temporary fixture.
        # Older sentence-transformers imported it despite trust_remote_code=False.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "imported.txt"
            (root / "config_sentence_transformers.json").write_text("{}")
            (root / "modules.json").write_text(json.dumps([
                {"idx": 0, "name": "0", "path": "", "type": "custom_module.CustomModule"}
            ]))
            (root / "custom_module.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('imported')\n"
                "class CustomModule:\n"
                "    @staticmethod\n"
                "    def load(*args, **kwargs):\n"
                "        raise RuntimeError('fixture reached custom loader')\n"
            )
            caught = None
            try:
                load_sentence_transformer(str(root))
            except Exception as exc:
                caught = exc
            self.assertFalse(marker.exists(), "Untrusted local module code executed")
            self.assertIsNotNone(caught, "Custom code must be rejected")
            self.assertIn("trust_remote_code", str(caught))

    def test_builtin_qwen3_model_encodes_without_custom_code(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model
        from sentence_transformers import SentenceTransformer, models

        torch.set_num_threads(2)
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            base.mkdir()
            tokenizer = Tokenizer(WordLevel(
                {"[UNK]": 0, "[PAD]": 1, "ego": 2, "yields": 3, "pedestrian": 4},
                unk_token="[UNK]",
            ))
            tokenizer.pre_tokenizer = Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                   pad_token="[PAD]").save_pretrained(base)
            Qwen3Model(Qwen3Config(vocab_size=5, hidden_size=32, intermediate_size=64,
                       num_hidden_layers=1, num_attention_heads=4,
                       num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
                       pad_token_id=1)).save_pretrained(base)
            transformer = models.Transformer(str(base), max_seq_length=16)
            model = SentenceTransformer(modules=[transformer, models.Pooling(
                32, pooling_mode="lasttoken")], device="cpu")
            texts = ["ego yields", "pedestrian"]
            expected = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            saved = root / "saved"
            model.save(str(saved))
            loaded = load_sentence_transformer(str(saved))
            actual = loaded.encode(texts, batch_size=2, normalize_embeddings=True,
                                   show_progress_bar=False, convert_to_numpy=True)
            self.assertEqual(actual.shape, (2, 32))
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-6)
            np.testing.assert_allclose(actual, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
