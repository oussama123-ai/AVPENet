"""
Unit tests for AVPENet components.

Run with: pytest tests/ -v
"""

import pytest
import torch
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Audio Encoder
# ──────────────────────────────────────────────────────────────────────────────

class TestAudioEncoder:
    def test_output_shape(self):
        from avpenet.models.audio_encoder import AudioEncoder
        model = AudioEncoder(embed_dim=512, pretrained=False)
        x = torch.randn(4, 1, 128, 300)
        ea = model(x)
        assert ea.shape == (4, 512), f"Expected (4, 512), got {ea.shape}"

    def test_custom_embed_dim(self):
        from avpenet.models.audio_encoder import AudioEncoder
        model = AudioEncoder(embed_dim=256, pretrained=False)
        x = torch.randn(2, 1, 128, 300)
        ea = model(x)
        assert ea.shape == (2, 256)

    def test_single_sample(self):
        from avpenet.models.audio_encoder import AudioEncoder
        model = AudioEncoder(pretrained=False)
        x = torch.randn(1, 1, 128, 300)
        ea = model(x)
        assert ea.shape == (1, 512)

    def test_no_nan(self):
        from avpenet.models.audio_encoder import AudioEncoder
        model = AudioEncoder(pretrained=False)
        x = torch.randn(4, 1, 128, 300)
        ea = model(x)
        assert not torch.isnan(ea).any()


# ──────────────────────────────────────────────────────────────────────────────
# Visual Encoder
# ──────────────────────────────────────────────────────────────────────────────

class TestVisualEncoder:
    def test_output_shape(self):
        from avpenet.models.visual_encoder import VisualEncoder
        model = VisualEncoder(embed_dim=512, pretrained=False)
        x = torch.randn(4, 3, 224, 224)
        ev = model(x)
        assert ev.shape == (4, 512)

    def test_spatial_attention_shape(self):
        from avpenet.models.visual_encoder import SpatialAttentionModule
        attn = SpatialAttentionModule(kernel_size=7)
        x = torch.randn(2, 2048, 7, 7)
        out = attn(x)
        assert out.shape == x.shape

    def test_no_nan(self):
        from avpenet.models.visual_encoder import VisualEncoder
        model = VisualEncoder(pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        ev = model(x)
        assert not torch.isnan(ev).any()


# ──────────────────────────────────────────────────────────────────────────────
# Cross-Modal Fusion
# ──────────────────────────────────────────────────────────────────────────────

class TestCrossModalFusion:
    def test_output_shape(self):
        from avpenet.models.fusion import CrossModalFusion
        fusion = CrossModalFusion(embed_dim=512, num_heads=8)
        ea = torch.randn(4, 512)
        ev = torch.randn(4, 512)
        f_fused, _ = fusion(ea, ev)
        assert f_fused.shape == (4, 512)

    def test_attention_weights_returned(self):
        from avpenet.models.fusion import CrossModalFusion
        fusion = CrossModalFusion(embed_dim=512, num_heads=8)
        ea = torch.randn(2, 512)
        ev = torch.randn(2, 512)
        _, attn = fusion(ea, ev)
        assert "audio_to_visual" in attn
        assert "visual_to_audio" in attn

    def test_bidirectional_different(self):
        """audio→visual and visual→audio outputs should differ."""
        from avpenet.models.fusion import CrossModalFusion
        torch.manual_seed(0)
        fusion = CrossModalFusion(embed_dim=64, num_heads=4)
        ea = torch.randn(2, 64)
        ev = torch.randn(2, 64) * 2   # different scale
        f1, attn = fusion(ea, ev)
        # Swap
        f2, _ = fusion(ev, ea)
        # They should differ (different input → different output)
        assert not torch.allclose(f1, f2)

    def test_no_nan(self):
        from avpenet.models.fusion import CrossModalFusion
        fusion = CrossModalFusion(embed_dim=512, num_heads=8)
        ea = torch.randn(4, 512)
        ev = torch.randn(4, 512)
        f, _ = fusion(ea, ev)
        assert not torch.isnan(f).any()


# ──────────────────────────────────────────────────────────────────────────────
# Regression Head
# ──────────────────────────────────────────────────────────────────────────────

class TestRegressionHead:
    def test_output_shape(self):
        from avpenet.models.regression_head import RegressionHead
        head = RegressionHead(embed_dim=512)
        x = torch.randn(8, 512)
        y = head(x)
        assert y.shape == (8,)

    def test_output_range(self):
        """All outputs should be in [0, 10]."""
        from avpenet.models.regression_head import RegressionHead
        head = RegressionHead(embed_dim=512)
        x = torch.randn(100, 512)
        y = head(x)
        assert y.min() >= 0.0
        assert y.max() <= 10.0

    def test_no_nan(self):
        from avpenet.models.regression_head import RegressionHead
        head = RegressionHead()
        x = torch.randn(4, 512)
        y = head(x)
        assert not torch.isnan(y).any()


# ──────────────────────────────────────────────────────────────────────────────
# Full AVPENet
# ──────────────────────────────────────────────────────────────────────────────

class TestAVPENet:
    @pytest.fixture
    def model(self):
        from avpenet.models.avpenet import AVPENet
        return AVPENet(pretrained=False)

    def test_forward_pass(self, model):
        audio  = torch.randn(2, 1, 128, 300)
        visual = torch.randn(2, 3, 224, 224)
        out = model(audio, visual)
        assert "pain_score" in out
        assert out["pain_score"].shape == (2,)

    def test_output_range(self, model):
        model.eval()
        with torch.no_grad():
            audio  = torch.randn(10, 1, 128, 300)
            visual = torch.randn(10, 3, 224, 224)
            out = model(audio, visual)
        assert out["pain_score"].min() >= 0.0
        assert out["pain_score"].max() <= 10.0

    def test_multiframe_visual(self, model):
        """Model should accept (B, T, 3, 224, 224) multi-frame input."""
        model.eval()
        with torch.no_grad():
            audio  = torch.randn(2, 1, 128, 300)
            visual = torch.randn(2, 30, 3, 224, 224)
            out = model(audio, visual)
        assert out["pain_score"].shape == (2,)

    def test_return_embeddings(self, model):
        model.eval()
        with torch.no_grad():
            audio  = torch.randn(2, 1, 128, 300)
            visual = torch.randn(2, 3, 224, 224)
            out = model(audio, visual, return_embeddings=True)
        assert "ea" in out
        assert "ev" in out
        assert out["ea"].shape == (2, 512)

    def test_return_attention(self, model):
        model.eval()
        with torch.no_grad():
            audio  = torch.randn(2, 1, 128, 300)
            visual = torch.randn(2, 3, 224, 224)
            out = model(audio, visual, return_attention=True)
        assert "attn_weights" in out

    def test_freeze_unfreeze(self, model):
        model.freeze_encoders()
        for p in model.audio_encoder.parameters():
            assert not p.requires_grad
        model.unfreeze_encoders()
        for p in model.audio_encoder.parameters():
            assert p.requires_grad

    def test_parameter_groups(self, model):
        groups = model.get_parameter_groups(lr_encoder=1e-5, lr_fusion=1e-4)
        assert len(groups) == 2
        assert groups[0]["lr"] == 1e-5
        assert groups[1]["lr"] == 1e-4

    def test_no_nan_forward(self, model):
        model.eval()
        with torch.no_grad():
            audio  = torch.randn(4, 1, 128, 300)
            visual = torch.randn(4, 3, 224, 224)
            out = model(audio, visual)
        assert not torch.isnan(out["pain_score"]).any()


# ──────────────────────────────────────────────────────────────────────────────
# Loss Functions
# ──────────────────────────────────────────────────────────────────────────────

class TestLosses:
    def test_mse_loss(self):
        from avpenet.losses import MSELoss
        loss_fn = MSELoss()
        pred   = torch.tensor([3.0, 5.0, 7.0])
        target = torch.tensor([3.0, 5.0, 7.0])
        assert loss_fn(pred, target).item() == pytest.approx(0.0)

    def test_ordinal_loss_correct_order(self):
        """Perfectly ordered predictions should have near-zero ordinal loss."""
        from avpenet.losses import OrdinalConsistencyLoss
        loss_fn = OrdinalConsistencyLoss(margin=0.5)
        pred   = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
        target = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
        assert loss_fn(pred, target).item() == pytest.approx(0.0)

    def test_boundary_loss_in_range(self):
        """Predictions in [1, 9] should have zero smoothness loss."""
        from avpenet.losses import BoundarySmoothnessLoss
        loss_fn = BoundarySmoothnessLoss()
        pred = torch.tensor([1.0, 5.0, 9.0])
        assert loss_fn(pred).item() == pytest.approx(0.0)

    def test_composite_loss_keys(self):
        from avpenet.losses import CompositePainLoss
        loss_fn = CompositePainLoss()
        pred   = torch.tensor([3.0, 5.0, 7.0])
        target = torch.tensor([3.5, 4.5, 7.5])
        result = loss_fn(pred, target)
        assert "total" in result
        assert "mse" in result
        assert "ordinal" in result
        assert "smooth" in result

    def test_composite_loss_positive(self):
        from avpenet.losses import CompositePainLoss
        loss_fn = CompositePainLoss()
        pred   = torch.randn(8).clamp(0, 10)
        target = torch.randn(8).clamp(0, 10)
        result = loss_fn(pred, target)
        assert result["total"].item() >= 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_mae_zero(self):
        from avpenet.metrics import compute_mae
        pred   = np.array([1.0, 5.0, 9.0])
        target = np.array([1.0, 5.0, 9.0])
        assert compute_mae(pred, target) == pytest.approx(0.0)

    def test_pcc_perfect(self):
        from avpenet.metrics import compute_pcc
        x = np.arange(10, dtype=float)
        assert compute_pcc(x, x) == pytest.approx(1.0)

    def test_discretise_pain(self):
        from avpenet.metrics import discretise_pain
        scores = np.array([0, 1.5, 3.0, 3.5, 6.0, 6.5, 10.0])
        labels = discretise_pain(scores)
        assert labels[0] == 0   # low
        assert labels[3] == 1   # moderate
        assert labels[6] == 2   # high

    def test_evaluate_returns_all_keys(self):
        from avpenet.metrics import evaluate
        pred   = np.random.uniform(0, 10, 50)
        target = np.random.uniform(0, 10, 50)
        result = evaluate(pred, target)
        for key in ["mae", "rmse", "pcc", "icc", "accuracy", "f1", "kappa"]:
            assert key in result

    def test_evaluate_age_stratified(self):
        from avpenet.metrics import evaluate
        np.random.seed(42)
        pred   = np.random.uniform(0, 10, 100)
        target = np.random.uniform(0, 10, 100)
        groups = np.array(["neonate"] * 50 + ["adult"] * 50)
        result = evaluate(pred, target, groups)
        assert "neonate" in result
        assert "adult"   in result
        assert result["neonate"]["n"] == 50
        assert result["adult"]["n"]   == 50
