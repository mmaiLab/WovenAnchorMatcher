"""Opt-in anchor assignment probability capture and statistics; no model parameters."""

import torch


class AnchorProbabilityAnalysis:
    """Store detached probabilities only when visualization/statistics is enabled."""

    def __init__(self, n_anchors, window_size, use_overlap):
        self.n_anchors = n_anchors
        self.window_size = window_size
        self.use_overlap = use_overlap
        self.capture_anchor_probs = False
        self.capture_anchor_stats = False
        self._anchor_probs_cache = None
        self._anchor_stats_cache = None

    def set_capture(self, enabled: bool):
        self.capture_anchor_probs = bool(enabled)
        if not self.capture_anchor_probs:
            self._anchor_probs_cache = None

    def set_statistics(self, enabled: bool):
        self.capture_anchor_stats = bool(enabled)
        if not self.capture_anchor_stats:
            self._anchor_stats_cache = None

    def cache_probabilities(
        self,
        assign_probs0,
        original_hw0,
        pad_hw0,
        assign_probs1,
        original_hw1,
        pad_hw1,
    ):
        if not self.capture_anchor_probs:
            return

        self._anchor_probs_cache = {
            "image0": {
                "assign_probs": assign_probs0.detach(),
                "original_hw": tuple(original_hw0),
                "pad_hw": tuple(pad_hw0),
            },
            "image1": {
                "assign_probs": assign_probs1.detach(),
                "original_hw": tuple(original_hw1),
                "pad_hw": tuple(pad_hw1),
            },
        }

    def cache_statistics(self, *probability_tensors):
        """Accumulate P_{m,k} and P_{m,k}^2 over valid token-window entries for each k.

        Overlapping occurrences count separately, as in the paper's assignment analysis.
        """
        if not self.capture_anchor_stats:
            return

        anchor_prob_sum = None
        anchor_prob_sq_sum = None
        valid_count = None
        for assign_probs in probability_tensors:
            probs_double = assign_probs.detach().double()
            current_sum = probs_double.sum(dim=(0, 2, 3))
            current_sum_sq = probs_double.square().sum(dim=(0, 2, 3))
            current_count = (probs_double.sum(dim=1) > 0).sum()

            if anchor_prob_sum is None:
                anchor_prob_sum = current_sum
                anchor_prob_sq_sum = current_sum_sq
                valid_count = current_count
            else:
                anchor_prob_sum = anchor_prob_sum + current_sum
                anchor_prob_sq_sum = anchor_prob_sq_sum + current_sum_sq
                valid_count = valid_count + current_count

        self._anchor_stats_cache = {
            "sum": anchor_prob_sum,
            "sum_sq": anchor_prob_sq_sum,
            "count": valid_count,
        }

    def pop_statistics(self):
        cache = self._anchor_stats_cache
        self._anchor_stats_cache = None
        if cache is None:
            return None

        return {
            "n_anchors": self.n_anchors,
            "sum": cache["sum"].double().cpu().tolist(),
            "sum_sq": cache["sum_sq"].double().cpu().tolist(),
            "count": int(cache["count"].item()),
        }

    def _make_offset_views(self, assign_probs, original_hw, pad_hw):
        """Group assignment maps by window origin offset on the original grid."""
        H0, W0 = (int(v) for v in original_hw)
        pad_h, pad_w = (int(v) for v in pad_hw)
        B, K, H_u, W_u = assign_probs.shape
        w = self.window_size

        # A positive sum over anchor IDs marks valid positions;
        # masked positions have zero assignments.
        valid_unfolded = assign_probs.sum(dim=1, keepdim=True) > 0

        def place_on_original(name, offset, probs_src, valid_src):
            y0, x0 = offset
            height = min(probs_src.shape[-2], max(H0 - y0, 0))
            width = min(probs_src.shape[-1], max(W0 - x0, 0))

            probs = probs_src.new_zeros(B, K, H0, W0)
            valid = torch.zeros(
                B, 1, H0, W0, dtype=torch.bool, device=probs_src.device
            )
            if height > 0 and width > 0:
                probs[:, :, y0:y0 + height, x0:x0 + width] = probs_src[:, :, :height, :width]
                valid[:, :, y0:y0 + height, x0:x0 + width] = valid_src[:, :, :height, :width]

            return {
                "name": name,
                "offset": (y0, x0),
                "extent": (y0, y0 + height, x0, x0 + width),
                "window_size": w,
                "probs": probs.detach().float().cpu(),
                "valid": valid.detach().cpu(),
            }

        if not self.use_overlap:
            return [place_on_original("base", (0, 0), assign_probs, valid_unfolded)]

        s = w // 2
        H = H0 + pad_h
        W = W0 + pad_w
        h_top = (H // w) * w
        h_bottom = ((H - s) // w) * w
        w_left = (W // w) * w
        w_right = ((W - s) // w) * w

        expected_hw = (h_top + h_bottom, w_left + w_right)
        if (H_u, W_u) != expected_hw:
            raise RuntimeError(
                f"Unexpected unfolded shape {(H_u, W_u)}; expected {expected_hw}."
            )

        offset_specs = [
            ("lt", (0, 0), assign_probs[:, :, :h_top, :w_left], valid_unfolded[:, :, :h_top, :w_left]),
            ("rt", (0, s), assign_probs[:, :, :h_top, w_left:], valid_unfolded[:, :, :h_top, w_left:]),
            ("lb", (s, 0), assign_probs[:, :, h_top:, :w_left], valid_unfolded[:, :, h_top:, :w_left]),
            ("rb", (s, s), assign_probs[:, :, h_top:, w_left:], valid_unfolded[:, :, h_top:, w_left:]),
        ]
        return [place_on_original(*spec) for spec in offset_specs]

    def pop_views(self):
        cache = self._anchor_probs_cache
        self._anchor_probs_cache = None
        if cache is None:
            return None

        image0 = cache["image0"]
        image1 = cache["image1"]
        return {
            "n_anchors": self.n_anchors,
            "window_size": self.window_size,
            "use_overlap": self.use_overlap,
            "image0": self._make_offset_views(**image0),
            "image1": self._make_offset_views(**image1),
        }
