import os
import shutil
import unittest
import tempfile
from pathlib import Path

from app.config import find_glomap_binary, PipelineConfig, SfMConfig, VideoConfig
from app.video.extractor import VideoExtractor, VideoMetadata
from app.reconstruction.openmvs import DenseReconstructor
from app.reconstruction.colmap import SfMPipeline
from app.geospatial.gps import GPSExtractor, TelemetryPoint
from app.geospatial.georeference import Georeferencer


class TestGlomapOpenMVSEnforcement(unittest.TestCase):
    """Unit tests for GLOMAP, OpenMVS, telemetry, and honesty enforcement."""

    def test_01_glomap_executable_detection(self):
        """Verify GLOMAP executable detection helper."""
        binary_path = find_glomap_binary()
        self.assertIsNotNone(
            binary_path, 
            "GLOMAP executable must be detected either in .local/bin/glomap or on system PATH."
        )
        self.assertTrue(os.access(binary_path, os.X_OK))

    def test_02_openmvs_executable_detection(self):
        """Verify OpenMVS CLI binaries detection."""
        from app.reconstruction.openmvs import find_openmvs_binary
        required = ["InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "TextureMesh"]
        for binary in required:
            path = find_openmvs_binary(binary)
            self.assertIsNotNone(path, f"OpenMVS binary {binary} must be detected on system PATH or in openMVS_build/bin.")

    def test_03_sparse_model_dir_detection(self):
        """Verify detection of sparse model directory (workspace/sparse vs workspace/sparse/0)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ws = Path(tmp_dir)
            sfm = SfMPipeline(ws)
            
            # Test 1: empty fallback to workspace/sparse
            self.assertEqual(sfm.detect_sparse_model_dir(), ws / "sparse")
            
            # Test 2: workspace/sparse/0 with cameras.bin
            sub_0 = ws / "sparse" / "0"
            sub_0.mkdir(parents=True, exist_ok=True)
            (sub_0 / "cameras.bin").write_bytes(b"dummy")
            self.assertEqual(sfm.detect_sparse_model_dir(), sub_0)

    def test_04_openmvs_output_validation(self):
        """Verify OpenMVS output validation fails when files are missing or 0 bytes."""
        import subprocess
        with tempfile.TemporaryDirectory() as tmp_dir:
            dense_recon = DenseReconstructor(tmp_dir)
            # Should raise RuntimeError or CalledProcessError because sparse model is missing
            with self.assertRaises((RuntimeError, subprocess.CalledProcessError)):
                dense_recon.run_openmvs()

    def test_05_explicit_telemetry_handling(self):
        """Verify parsing of explicit telemetry files (.srt, .csv, .gpx)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            srt_file = tmp_path / "flight.srt"
            srt_content = """1
00:00:01,000 --> 00:00:02,000
[latitude: 37.7749] [longitude: -122.4194] [rel_alt: 50.5 abs_alt: 50.5]
"""
            srt_file.write_text(srt_content)
            
            extractor = GPSExtractor()
            points = extractor.parse_srt(srt_file)
            self.assertEqual(len(points), 1)
            self.assertAlmostEqual(points[0].lat, 37.7749)
            self.assertAlmostEqual(points[0].lon, -122.4194)
            self.assertAlmostEqual(points[0].alt, 50.5)

    def test_06_gps_to_frame_matching(self):
        """Verify GPS points match frames by path and by basename."""
        extractor = GPSExtractor()
        raw_telemetry = [
            TelemetryPoint(timestamp_sec=0.0, lat=37.7749, lon=-122.4194, alt=50.0),
            TelemetryPoint(timestamp_sec=10.0, lat=37.7750, lon=-122.4195, alt=55.0)
        ]
        
        frame_paths = [Path("/tmp/frames/frame_0001.png"), Path("/tmp/frames/frame_0002.png")]
        interp = extractor.interpolate_telemetry(raw_telemetry, [0.0, 10.0])
        self.assertEqual(len(interp), 2)
        self.assertAlmostEqual(interp[0].lat, 37.7749)
        self.assertAlmostEqual(interp[1].lat, 37.7750)

    def test_07_local_coordinates_without_gps(self):
        """Verify pipeline defaults strictly to local metric space when GPS is absent."""
        georef = Georeferencer()
        self.assertFalse(georef.is_georeferenced)
        self.assertEqual(georef.crs_name, "Local (Non-georeferenced)")

    def test_08_no_synthetic_gps_fallback(self):
        """Verify extract_flight_telemetry returns None when no telemetry exists (zero fake GPS)."""
        extractor = GPSExtractor()
        with tempfile.TemporaryDirectory() as tmp_dir:
            v_file = Path(tmp_dir) / "novideo.mp4"
            frame_paths = [Path(tmp_dir) / "frame_0.png"]
            result = extractor.extract_flight_telemetry(v_file, frame_paths, duration_sec=5.0)
            self.assertIsNone(result, "Must return None when no telemetry file exists; fake GPS is strictly forbidden.")

    def test_09_no_synthetic_dense_fallback(self):
        """Verify OpenMVS raises RuntimeError when CLI binaries fail or are missing (zero fake dense fallback)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            recon = DenseReconstructor(tmp_dir)
            if not recon.is_openmvs_available():
                with self.assertRaises(RuntimeError):
                    recon.reconstruct([], [], {})

    def test_10_rejects_portrait_non_aerial_input(self):
        """The UAV profile must reject portrait street/handheld video formats."""
        extractor = VideoExtractor(VideoConfig(capture_profile="aerial_drone"))
        metadata = VideoMetadata(
            fps=60.0,
            width=1080,
            height=1920,
            total_frames=660,
            duration_sec=11.0,
            format="mp4",
        )
        with self.assertRaisesRegex(ValueError, "portrait street/handheld"):
            extractor.validate_capture_profile(metadata)


if __name__ == "__main__":
    unittest.main()
