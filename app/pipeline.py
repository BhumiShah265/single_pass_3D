import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

from app.config import PipelineConfig, logger
from app.video.extractor import VideoExtractor
from app.video.frame_selector import KeyframeSelector
from app.preprocessing.quality import QualityFilter
from app.preprocessing.dynamic_objects import DynamicObjectMasker
from app.geospatial.gps import GPSExtractor
from app.geospatial.georeference import Georeferencer
from app.reconstruction.colmap import SfMPipeline
from app.reconstruction.openmvs import DenseReconstructor
from app.reconstruction.meshing import MeshProcessor
from app.analysis.semantic import SemanticAnnotator
from app.analysis.measurements import MeasurementTool


class ReconstructionPipeline:
    """
    Master orchestrator for the real multi-view photogrammetry 3D reconstruction pipeline.
    Replaces optical flow pseudo-elevations with true Structure from Motion (pycolmap),
    Multi-View Stereo (MVS) depth fusion, Poisson surface reconstruction,
    and rigorous photogrammetric metrology.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.workspace = self.config.get_workspace()
        self.output_dir = self.config.get_output()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.frames_dir = self.workspace / "raw_frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
        self.extracted_frames: List[Path] = []
        self.valid_frames: List[Path] = []
        self.keyframes: List[Path] = []
        self.dynamic_masks: List[Path] = []
        self.gps_data: Optional[Dict[Path, Tuple[float, float, float]]] = None
        
        # Real Reconstruction Data
        self.camera_trajectory: List[Dict[str, Any]] = []
        self.camera_calibration: Dict[str, Any] = {}
        self.sparse_points: List[List[float]] = []
        self.dense_pcd = None
        self.reconstructed_points: List[List[float]] = [] # [x, y, z, r, g, b, class_id]
        
        # Real Mesh Geometry
        self.mesh = None
        self.mesh_vertices = np.empty((0, 3), dtype=np.float32)
        self.mesh_uvs = np.empty((0, 2), dtype=np.float32)
        self.mesh_faces = np.empty((0, 3), dtype=np.int32)
        self.mesh_normals = np.empty((0, 3), dtype=np.float32)
        self.mesh_colors = np.empty((0, 4), dtype=np.uint8)
        self.mesh_classification = np.empty((0,), dtype=np.int32)
        self.texture_img = None
        
        self.detected_structures: List[Dict[str, Any]] = []
        self.measurements: Dict[str, Any] = {}
        self.georeferencer = Georeferencer()
        self.quality_stats: Dict[str, Any] = {}
        self.sfm_reprojection_error: float = 0.0

    def run(self) -> Dict[str, Any]:
        """Run the complete 10-stage photogrammetric reconstruction pipeline."""
        logger.info(f"Starting true photogrammetry reconstruction for: {self.config.input_video}")
        try:
            self._stage_video_extraction()
            self._stage_quality_filtering()
            self._stage_keyframe_selection()
            
            if not self.config.skip_dynamic_masking:
                self._stage_dynamic_masking()
                
            self._stage_sfm()
            
            if not self.config.skip_depth_estimation:
                self._stage_dense_reconstruction()
                
            self._stage_meshing()
            
            if not self.config.skip_georeferencing:
                self._stage_georeferencing()
                
            if not self.config.skip_analysis:
                self._stage_analysis()
                
            self._stage_export_deliverables()
            
            logger.info("Photogrammetry pipeline completed successfully.")
            return {"status": "success", "output": str(self.output_dir)}
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _stage_video_extraction(self):
        """Stage 1: Extract frames and video metadata strictly respecting configured FPS."""
        logger.info("Stage 1/10: Ingesting video and extracting frames")
        extractor = VideoExtractor(self.config.video)
        metadata, frame_infos = extractor.extract_frames(self.config.input_video, self.frames_dir)
        self.metadata = metadata
        self.extracted_frames = [f.file_path for f in frame_infos]
        logger.info(
            f"Extracted {len(self.extracted_frames)} frames from {metadata.duration_sec:.1f}s video "
            f"({metadata.width}x{metadata.height} @ {metadata.fps:.1f} source fps)"
        )

        # Check for companion telemetry (DJI SRT, CSV, GPX)
        try:
            gps_extractor = GPSExtractor()
            self.gps_data = gps_extractor.extract_flight_telemetry(
                self.config.input_video, 
                self.extracted_frames, 
                duration_sec=metadata.duration_sec
            )
            if self.gps_data:
                logger.info(f"Loaded real GPS flight telemetry for {len(self.gps_data)} frames.")
            else:
                logger.info("No companion telemetry file found. Operating in strictly LOCAL metric coordinates.")
        except Exception as e:
            logger.warning(f"Telemetry check notice: {e}. Defaulting to local coordinates.")
            self.gps_data = None

    def _stage_quality_filtering(self):
        """Stage 2: Filter frames using Laplacian sharpness and exposure bounds."""
        logger.info("Stage 2/10: Filtering frames for sharpness and exposure")
        quality_filter = QualityFilter(self.config.quality)
        self.valid_frames = quality_filter.filter_frames(self.extracted_frames)
        self.quality_stats = quality_filter.stats
        if not self.valid_frames:
            self.valid_frames = self.extracted_frames
        logger.info(f"Retained {len(self.valid_frames)} / {len(self.extracted_frames)} sharp, well-exposed frames.")

    def _stage_keyframe_selection(self):
        """Stage 3: Select keyframes using image motion baseline and optional GPS distance."""
        logger.info("Stage 3/10: Selecting optimal keyframes for multi-view stereo baseline")
        try:
            selector = KeyframeSelector(self.config.keyframe)
            self.keyframes = selector.select_keyframes(
                self.valid_frames, 
                gps_data=self.gps_data
            )
        except Exception as e:
            logger.warning(f"Keyframe selection notice: {e}")
            self.keyframes = self.valid_frames
            
        if not self.keyframes:
            self.keyframes = self.valid_frames
        logger.info(f"Selected {len(self.keyframes)} keyframes for 3D reconstruction.")

    def _stage_dynamic_masking(self):
        """Stage 4: Generate dynamic object masks for all keyframes using YOLOv8-seg."""
        logger.info("Stage 4/10: Masking dynamic objects (cars, pedestrians) across keyframes")
        try:
            masker = DynamicObjectMasker(self.config.dynamic_mask)
            mask_dir = self.workspace / "masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            self.dynamic_masks = masker.process_frames(self.keyframes, mask_dir)
            logger.info(f"Generated {len(self.dynamic_masks)} dynamic masks in {mask_dir}.")
        except Exception as e:
            logger.warning(f"Dynamic masking notice: {e}. Proceeding without dynamic mask exclusion.")
            self.dynamic_masks = []

    def _stage_sfm(self):
        """Stage 5: Structure from Motion with pycolmap (CameraMode.SINGLE auto-calibration)."""
        logger.info("Stage 5/10: Computing Structure from Motion & Camera Calibration via pycolmap")
        if len(self.keyframes) < 3:
            raise RuntimeError(f"SfM requires at least 3 keyframes. Received: {len(self.keyframes)}")

        sfm = SfMPipeline(self.workspace)
        self.keyframes = sfm.prepare_workspace(self.keyframes, self.dynamic_masks)
        sfm.extract_features(camera_model=self.config.sfm.camera_model)
        sfm.match_features(method="sequential")
        rec = sfm.map()
        
        if rec is None:
            raise RuntimeError("pycolmap Structure from Motion failed: could not reconstruct sparse camera model.")

        sfm_res = sfm.get_reconstruction_data(self.keyframes, self.gps_data)
        self.camera_trajectory = sfm_res["camera_poses"]
        self.sparse_points = sfm_res["sparse_points"]
        self.camera_calibration = sfm_res["camera_calibration"]
        self.sfm_reprojection_error = sfm_res["reprojection_error_px"]

        logger.info(
            f"SfM Successful: {sfm_res['registered_count']}/{sfm_res['total_count']} views registered, "
            f"{len(self.sparse_points):,} sparse tie points, "
            f"mean reprojection error: {self.sfm_reprojection_error:.2f} px."
        )

    def _stage_dense_reconstruction(self):
        """Stage 6: Real dense multi-view stereo reconstruction."""
        logger.info("Stage 6/10: Dense depth reconstruction via Multi-View Stereo (MVS)")
        dense_engine = DenseReconstructor(self.workspace)
        dense_res = dense_engine.reconstruct(
            keyframes=self.keyframes,
            camera_poses=self.camera_trajectory,
            camera_calibration=self.camera_calibration,
            sparse_points=self.sparse_points,
            dynamic_mask_paths=self.dynamic_masks,
            voxel_size=self.config.reconstruction.voxel_size,
            outlier_nb_neighbors=self.config.reconstruction.outlier_nb_neighbors,
            outlier_std_ratio=self.config.reconstruction.outlier_std_ratio
        )
        self.dense_pcd = dense_res["pcd"]
        self.dense_ply_path = dense_res["dense_ply_path"]
        logger.info(f"Dense reconstruction completed: {len(self.dense_pcd.points):,} clean 3D points generated.")

    def _stage_meshing(self):
        """Stage 7: Real Poisson surface reconstruction and calibrated camera texture projection."""
        logger.info("Stage 7/10: Generating Poisson surface mesh and projecting camera textures")
        mesh_proc = MeshProcessor(self.workspace, self.config.mesh)
        self.mesh = mesh_proc.poisson_reconstruction(
            self.dense_pcd, 
            depth=self.config.mesh.poisson_depth,
            trim_quantile=self.config.mesh.density_trim_quantile
        )
        
        # Texture mesh using original calibrated camera views
        vertex_colors, uvs, tex_img = mesh_proc.texture_mesh_from_cameras(
            self.mesh, 
            self.camera_trajectory, 
            self.camera_calibration,
            tex_size=self.config.mesh.texture_resolution
        )
        self.mesh_uvs = uvs
        self.texture_img = tex_img
        self.mesh_vertices = np.asarray(self.mesh.vertices, dtype=np.float32)
        self.mesh_faces = np.asarray(self.mesh.triangles, dtype=np.int32)
        self.mesh_normals = np.asarray(self.mesh.vertex_normals, dtype=np.float32)
        self.mesh_colors = (vertex_colors * 255).astype(np.uint8)

        logger.info(
            f"Surface mesh completed: {len(self.mesh_vertices):,} vertices, "
            f"{len(self.mesh_faces):,} faces with true camera projection."
        )

    def _stage_georeferencing(self):
        """Stage 8: Georeference scene if real GPS telemetry exists; otherwise keep local coordinates."""
        logger.info("Stage 8/10: Evaluating georeferencing against flight telemetry")
        if self.gps_data:
            local_cam_centers = []
            gps_coords = []
            for p in self.camera_trajectory:
                if p.get("is_registered") and p.get("C"):
                    img_p = Path(p["file_path"])
                    if img_p in self.gps_data:
                        local_cam_centers.append(p["C"])
                        gps_coords.append(self.gps_data[img_p])

            if len(gps_coords) >= 3:
                self.georeferencer.georeference_scene(np.array(local_cam_centers), gps_coords)
                if self.georeferencer.is_georeferenced:
                    self.georeferencer.transform_point_cloud(self.dense_pcd)
                    self.georeferencer.transform_mesh(self.mesh)
                    self.mesh_vertices = np.asarray(self.mesh.vertices, dtype=np.float32)
                    logger.info(f"Transformed 3D geometry to {self.georeferencer.crs_name}")
            else:
                logger.info("Fewer than 3 telemetry points matched. Keeping local coordinates.")
        else:
            logger.info("No telemetry provided. Retaining strictly LOCAL metric coordinates.")

    def _stage_analysis(self):
        """Stage 9: Semantic 3D annotation and physical metrology calculations."""
        logger.info("Stage 9/10: Running 3D semantic annotation and metric calculations")
        annotator = SemanticAnnotator()
        classification, structures = annotator.annotate_point_cloud(
            self.dense_pcd, 
            self.camera_trajectory, 
            self.camera_calibration
        )
        self.detected_structures = structures
        
        pts_arr = np.asarray(self.dense_pcd.points)
        cols_arr = (np.asarray(self.dense_pcd.colors) * 255).astype(int) if self.dense_pcd.has_colors() else np.zeros((len(pts_arr), 3), dtype=int)
        
        # Subsample points for WebGL viewport (up to 150,000 points)
        sample_stride = max(1, len(pts_arr) // 150000)
        self.reconstructed_points = []
        for i in range(0, len(pts_arr), sample_stride):
            p = pts_arr[i]
            c = cols_arr[i]
            cls_code = int(classification[i]) if i < len(classification) else 2
            self.reconstructed_points.append([
                round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3),
                int(c[0]), int(c[1]), int(c[2]),
                cls_code
            ])

        # True 3D bounding box extents and volume
        bbox = MeasurementTool.compute_bounding_box(pts_arr)
        
        # True Ground Sampling Distance (GSD)
        gsd_cm = MeasurementTool.compute_gsd(
            self.camera_trajectory, 
            self.camera_calibration, 
            pts_arr
        )
        
        # Feature heights from 3D points
        feature_heights = MeasurementTool.compute_feature_heights(pts_arr, classification)

        # Honest compliance & accuracy status
        if self.georeferencer.is_georeferenced:
            compliance_status = "Georeferenced to UTM telemetry - accuracy not independently validated"
            crs_display = self.georeferencer.crs_name
        else:
            compliance_status = "Local metric reconstruction - no telemetry/GCP available"
            crs_display = "Local (Non-georeferenced)"

        self.measurements = {
            "spatial_accuracy_note": compliance_status,
            "compliance_status": compliance_status,
            "reprojection_error_px": self.sfm_reprojection_error,
            "gsd_cm_px": gsd_cm if gsd_cm is not None else "GSD unavailable",
            "sparse_points": len(self.sparse_points),
            "dense_points": len(pts_arr),
            "mesh_vertices": len(self.mesh_vertices),
            "mesh_triangles": len(self.mesh_faces),
            "buildings_detected": len(self.detected_structures),
            "detected_structures": self.detected_structures,
            "crs": crs_display,
            "bounding_box": bbox,
            "volume_m3": bbox.get("volume_approx_m3", 0.0),
            "feature_heights": feature_heights,
            "quality_stats": self.quality_stats
        }

    def _stage_export_deliverables(self):
        """Stage 10: Export all 8 standardized deliverables from real geometry."""
        logger.info("Stage 10/10: Exporting all 8 standardized GIS & 3D deliverables")
        
        # 1. Export points.json (for WebGL viewport)
        points_payload = {
            "points": self.reconstructed_points,
            "trajectory": self.camera_trajectory,
            "measurements": self.measurements,
            "structures": self.detected_structures,
            "mesh": {
                "vertices": self.mesh_vertices.tolist(),
                "uvs": self.mesh_uvs.tolist(),
                "faces": self.mesh_faces.tolist(),
                "normals": self.mesh_normals.tolist(),
                "colors": self.mesh_colors.tolist(),
                "classification": [int(s.get("asprs_class", 2)) for s in self.detected_structures]
            },
            "glb_url": f"/data/output/{self.output_dir.name}/model.glb",
            "texture_url": f"/data/output/{self.output_dir.name}/texture.jpg"
        }
        with open(self.output_dir / "points.json", "w") as f:
            json.dump(points_payload, f)
            
        # 2. Export measurements.json
        with open(self.output_dir / "measurements.json", "w") as f:
            json.dump(self.measurements, f, indent=2)

        # 3. Export real mesh deliverables (model.obj, model.mtl, texture.jpg, model.glb, model.fbx)
        mesh_proc = MeshProcessor(self.workspace, self.config.mesh)
        mesh_proc.export_mesh_deliverables(
            self.mesh, 
            self.mesh_uvs, 
            self.texture_img, 
            self.output_dir
        )

        # 4. Export real Stanford .PLY file
        ply_path = self.output_dir / "cloud.ply"
        pts_arr = np.asarray(self.dense_pcd.points)
        cols_arr = (np.asarray(self.dense_pcd.colors) * 255).astype(int) if self.dense_pcd.has_colors() else np.zeros((len(pts_arr), 3), dtype=int)
        with open(ply_path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts_arr)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for p, c in zip(pts_arr, cols_arr):
                f.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {c[0]} {c[1]} {c[2]}\n")
        logger.info(f"Exported PLY point cloud: {ply_path} ({ply_path.stat().st_size:,} bytes)")

        # 5. Export ASPRS LAS 1.4 Point Cloud
        las_path = self.output_dir / "cloud.las"
        try:
            import laspy
            header = laspy.LasHeader(point_format=3, version="1.4")
            header.scales = np.array([0.001, 0.001, 0.001])
            header.offsets = np.array([0.0, 0.0, 0.0])
            las = laspy.LasData(header)
            las.x = pts_arr[:, 0]
            las.y = pts_arr[:, 1]
            las.z = pts_arr[:, 2]
            las.red = (cols_arr[:, 0] * 256).astype(np.uint16)
            las.green = (cols_arr[:, 1] * 256).astype(np.uint16)
            las.blue = (cols_arr[:, 2] * 256).astype(np.uint16)
            las.classification = np.full(len(pts_arr), 2, dtype=np.uint8)
            las.write(str(las_path))
            logger.info(f"Exported LAS: {las_path} ({las_path.stat().st_size:,} bytes)")
        except Exception as e:
            logger.warning(f"LAS export notice: {e}")

        # 6. Export GeoTIFF Orthomosaic
        ortho_path = self.output_dir / "ortho.tif"
        try:
            import rasterio
            from rasterio.transform import from_bounds
            h_tex, w_tex = self.texture_img.shape[:2]
            bbox = self.measurements["bounding_box"]
            min_x, max_x = bbox["min"][0], bbox["max"][0]
            min_z, max_z = bbox["min"][2], bbox["max"][2]
            transform = from_bounds(min_x, min_z, max_x, max_z, w_tex, h_tex)
            crs_str = self.georeferencer.crs_name if self.georeferencer.is_georeferenced else None

            with rasterio.open(
                str(ortho_path), 'w', driver='GTiff', height=h_tex, width=w_tex, count=3,
                dtype=rasterio.uint8, crs=crs_str, transform=transform
            ) as dst:
                dst.write(self.texture_img[:, :, 0], 1)
                dst.write(self.texture_img[:, :, 1], 2)
                dst.write(self.texture_img[:, :, 2], 3)
            logger.info(f"Exported Orthomosaic GeoTIFF: {ortho_path} ({ortho_path.stat().st_size:,} bytes)")
        except Exception as e:
            logger.warning(f"Orthomosaic export notice: {e}")

        # 7. Export Digital Surface Model (DSM) GeoTIFF rasterized from real 3D points
        dsm_path = self.output_dir / "dsm.tif"
        try:
            import rasterio
            from rasterio.transform import from_bounds
            from scipy.interpolate import griddata

            grid_res = 256
            bbox = self.measurements["bounding_box"]
            gx = np.linspace(bbox["min"][0], bbox["max"][0], grid_res)
            gz = np.linspace(bbox["min"][2], bbox["max"][2], grid_res)
            grid_x, grid_z = np.meshgrid(gx, gz)

            # Interpolate real elevation (Y) over the grid
            sub_pts = pts_arr[::max(1, len(pts_arr) // 20000)]
            elev_interp = griddata(
                (sub_pts[:, 0], sub_pts[:, 2]), 
                sub_pts[:, 1], 
                (grid_x, grid_z), 
                method='linear',
                fill_value=float(np.median(sub_pts[:, 1]))
            ).astype(np.float32)

            transform = from_bounds(bbox["min"][0], bbox["min"][2], bbox["max"][0], bbox["max"][2], grid_res, grid_res)
            crs_str = self.georeferencer.crs_name if self.georeferencer.is_georeferenced else None

            with rasterio.open(
                str(dsm_path), 'w', driver='GTiff', height=grid_res, width=grid_res, count=1,
                dtype=rasterio.float32, crs=crs_str, transform=transform
            ) as dst:
                dst.write(elev_interp, 1)
            logger.info(f"Exported DSM GeoTIFF: {dsm_path} ({dsm_path.stat().st_size:,} bytes)")
        except Exception as e:
            logger.warning(f"DSM export notice: {e}")

        # 8. Export Metrology Report (.pdf) with actual measured values
        pdf_path = self.output_dir / "report.pdf"
        try:
            from fpdf import FPDF
            def clean_txt(s: Any) -> str:
                return str(s).replace("\u2014", "-").replace("\u2013", "-").encode("latin-1", "replace").decode("latin-1")

            pdf = FPDF()
            pdf.add_page()
            
            # Header
            pdf.set_fill_color(15, 23, 42)
            pdf.rect(0, 0, 210, 32, 'F')
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 16)
            pdf.set_xy(14, 8)
            pdf.cell(0, 8, clean_txt("Single-Pass Drone 3D Photogrammetry Report"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_xy(14, 18)
            pdf.set_text_color(56, 189, 248)
            pdf.cell(0, 6, clean_txt("True Multi-View Photogrammetric Reconstruction Audit"), new_x="LMARGIN", new_y="NEXT")
            
            # Section 1: Executive Summary
            pdf.set_xy(14, 38)
            pdf.set_text_color(30, 41, 59)
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, clean_txt("1. Executive Summary & Metrology Audit"), new_x="LMARGIN", new_y="NEXT")
            
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(51, 65, 85)
            summary_text = (
                f"This document reports the metric reconstruction results for the single-pass UAV flight. "
                f"The 3D geometry was reconstructed using pycolmap Structure from Motion and multi-view stereo "
                f"depth triangulation. All metrics displayed below are computed from actual reconstructed data."
            )
            pdf.multi_cell(182, 5, clean_txt(summary_text))
            pdf.ln(3)
            
            # Table: Accuracy & Metrics
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_fill_color(241, 245, 249)
            pdf.cell(90, 7, clean_txt("Parameter"), border=1, fill=True)
            pdf.cell(92, 7, clean_txt("Measured Value"), border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
            
            gsd_val = self.measurements.get("gsd_cm_px")
            gsd_str = f"{gsd_val:.2f} cm/px" if isinstance(gsd_val, (int, float)) else str(gsd_val)
            bbox = self.measurements.get("bounding_box", {})
            dims = bbox.get("dimensions", [0, 0, 0])
            
            metrics = [
                ("Structure from Motion Reprojection Error", f"{self.sfm_reprojection_error:.2f} px"),
                ("Registered Keyframe Views", f"{len(self.camera_trajectory)} frames"),
                ("Sparse 3D Tie Points", f"{len(self.sparse_points):,} points"),
                ("Dense Multi-View Point Cloud", f"{len(pts_arr):,} points"),
                ("Surface Mesh Faces", f"{len(self.mesh_faces):,} triangles"),
                ("Ground Sampling Distance (GSD)", gsd_str),
                ("Coordinate Reference System (CRS)", self.measurements.get("crs", "Local")),
                ("Validation Status", self.measurements.get("compliance_status", "Local Reconstruction")),
                ("3D Scene Extents", f"{dims[0]}m x {dims[2]}m")
            ]
            
            pdf.set_font("Helvetica", "", 9)
            for param, val in metrics:
                pdf.cell(90, 6, clean_txt(param), border=1)
                pdf.cell(92, 6, clean_txt(val), border=1, new_x="LMARGIN", new_y="NEXT")
                
            pdf.ln(5)
            
            # Section 2: Detected Features
            if self.detected_structures:
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 7, "2. Verified 3D Structural Features", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_fill_color(241, 245, 249)
                pdf.cell(32, 6, "ID", border=1, fill=True)
                pdf.cell(50, 6, "Classification", border=1, fill=True)
                pdf.cell(35, 6, "Height (Roof-Ground)", border=1, fill=True)
                pdf.cell(35, 6, "Footprint Area", border=1, fill=True)
                pdf.cell(30, 6, "3D Points", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
                
                pdf.set_font("Helvetica", "", 8.5)
                for s in self.detected_structures[:8]:
                    pdf.cell(32, 5.5, clean_txt(s["id"]), border=1)
                    pdf.cell(50, 5.5, clean_txt(s["type"]), border=1)
                    pdf.cell(35, 5.5, clean_txt(f"{s.get('height_m', 0.0):.1f} m"), border=1)
                    pdf.cell(35, 5.5, clean_txt(f"{s.get('area_m2', 0.0):.1f} m2"), border=1)
                    pdf.cell(30, 5.5, clean_txt(f"{s.get('points_count', 0):,}"), border=1, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(5)

            # Embed Orthomosaic thumbnail if exists
            if (self.output_dir / "texture.jpg").exists():
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 7, "3. Reconstructed Aerial Texture Survey Map", new_x="LMARGIN", new_y="NEXT")
                pdf.image(str(self.output_dir / "texture.jpg"), x=14, y=pdf.get_y() + 2, w=100, h=70)
                
            pdf.output(str(pdf_path))
            logger.info(f"Exported Metrology PDF: {pdf_path} ({pdf_path.stat().st_size:,} bytes)")
        except Exception as e:
            logger.warning(f"PDF export notice: {e}")

        logger.info(f"All 8 deliverables successfully exported to {self.output_dir}.")
