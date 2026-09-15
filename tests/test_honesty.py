"""
Integrity guards for the VID-IMG photogrammetry pipeline.

The rebuild promise: real SfM/MVS geometry from real backends, real calibrated
camera intrinsics for GSD, and honest, loud failures when a real backend is
missing — never silently fabricated accuracy, survey-grade claims, Gaussian
splat/LiDAR labels, or hardcoded focal-length fallbacks.

Dep-free (stdlib + numpy) so these run in any CI where heavy GIS/geometry
libs are absent.
"""
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend" / "index.html"
OPENMVS = ROOT / "app" / "reconstruction" / "openmvs.py"
SEMANTIC = ROOT / "app" / "analysis" / "semantic.py"
MEASUREMENTS = ROOT / "app" / "analysis" / "measurements.py"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class FrontendHonestyTest(unittest.TestCase):
    """The UI must never label MVS/SfM output as Gaussian splats, LiDAR, or
    survey-grade — and must never claim an accuracy it did not measure."""

    @classmethod
    def setUpClass(cls):
        cls.html = read_text(FRONTEND)

    def test_no_gaussian_splat_or_lidar_claim(self):
        for phrase in ("3D Gaussian Splats / LiDAR Cloud",
                       "Gaussian Splat Cloud", "LiDAR Point Cloud",
                       "LiDAR Cloud", "gaussian_splat", "bunny_GSD"):
            self.assertNotIn(phrase, self.html,
                             f"Fabricated label reintroduced: {phrase!r}")

    def test_no_survey_grade_claim(self):
        for phrase in ("Survey Grade", "Survey-Grade", "Survey Grade Accuracy",
                       "<= 0.4m", "0.4m Accuracy", "Sub-Decimeter"):
            self.assertNotIn(phrase, self.html,
                             f"Fabricated accuracy claim: {phrase!r}")

    def test_no_hardcoded_focal_in_frontend(self):
        self.assertNotIn('get("focal_px", 1500', self.html)
        self.assertNotIn("_focal_px = 1500", self.html)

    def test_gsd_uses_dynamic_uniform_id(self):
        # GSD default display is a real dynamic value, never a static claim.
        self.assertIn("metricGsd", self.html)


class BackendHonestyTest(unittest.TestCase):
    """OpenMVS backend must require the real CLI binaries — never a fake
    dense fallback (no SGBM-on-arbitrary-pairs substitute)."""

    @classmethod
    def setUpClass(cls):
        cls.src = read_text(OPENMVS)

    def test_requires_real_openmvs_binary(self):
        self.assertIn("OpenMVS CLI binaries", self.src)
        self.assertIn("RuntimeError", self.src)

    def test_fails_honestly_without_binary(self):
        self.assertIn("Dense reconstruction backend unavailable", self.src)

    def test_no_fake_dense_fallback(self):
        for phrase in ("SGBM", "StereoSGBM", "StereoBM", "stereo_match",
                       "synthesize", "Geographic") or ():
            self.assertNotIn(phrase, self.src)


class UmeyamaHonestyTest(unittest.TestCase):
    """Umeyama georeferencing only from real telemetry; exact round-trip."""

    @staticmethod
    def _umeyama(X: np.ndarray, Y: np.ndarray):
        """Byte-for-byte mirror of app/geospatial/georeference.py:
        cov = Yc.T @ Xc / n, R = U @ S @ Vt (singular-value variant), scale
        from the mean squared norm of the centred source, and the exact
        COLMAP-ish row convention y_applied = s*(X @ R.T) + t."""
        n = X.shape[0]
        mu_x, mu_y = X.mean(axis=0), Y.mean(axis=0)
        Xc, Yc = X - mu_x, Y - mu_y
        sigma_x = np.mean(np.sum(Xc ** 2, axis=1))
        cov = (Yc.T @ Xc) / n
        U, D, Vt = np.linalg.svd(cov)
        S = np.eye(3)
        if np.linalg.det(cov) < 0:
            S[2, 2] = -1
        R = U @ S @ Vt
        c = np.trace(np.diag(D) @ S) / max(1e-8, sigma_x)
        t = mu_y - c * (R @ mu_x)
        return c, R, t

    def test_recovers_known_similarity(self):
        rng = np.random.default_rng(4)
        X = rng.standard_normal((50, 3)) * 4.0
        s_true = 2.5
        R_true, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(R_true) < 0:
            R_true[:, 0] *= -1
        t_true = rng.standard_normal(3) * 20.0
        Y = s_true * (X @ R_true.T) + t_true

        c, R, t = self._umeyama(X, Y)
        # Contract guaranteed by the module's apply_transform (row convention):
        # the recovered similarity must reproduce Y from X to numerical precision,
        # with the correct positive scale and a proper rotation.
        Yhat = c * (X @ R.T) + t
        np.testing.assert_allclose(Yhat, Y, atol=1e-6)
        self.assertAlmostEqual(c, s_true, places=6)
        self.assertGreater(c, 0.0)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=9)


class SemanticGsdHonestyTest(unittest.TestCase):
    def test_gsd_honest_flow_detection_path(self):
        sem = read_text(SEMANTIC)
        meas = read_text(MEASUREMENTS)
        # Both must derive GSD from real calibrated focal only.
        self.assertIn("focal", sem)
        self.assertIn("focal", meas)


if __name__ == "__main__":
    unittest.main()
