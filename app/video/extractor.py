import cv2
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from app.config import logger, VideoConfig

@dataclass
class VideoMetadata:
    """Structured metadata for a video file."""
    fps: float
    width: int
    height: int
    total_frames: int
    duration_sec: float
    format: str

@dataclass
class FrameInfo:
    """Structured information about an extracted frame."""
    frame_idx: int
    timestamp_sec: float
    file_path: Path

class VideoExtractor:
    """Extracts frames and metadata from video files."""

    def __init__(self, config: Optional[VideoConfig] = None):
        """
        Initialize the VideoExtractor.

        Args:
            config: Video extraction configuration.
        """
        self.config = config or VideoConfig()

    def get_metadata(self, video_path: str | Path) -> VideoMetadata:
        """
        Extract metadata from a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            VideoMetadata containing FPS, dimensions, total frames, etc.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Prevent division by zero if FPS is somehow 0
            duration_sec = total_frames / fps if fps > 0 else 0.0
            
            format_ext = video_path.suffix.lower().strip('.')

            return VideoMetadata(
                fps=fps,
                width=width,
                height=height,
                total_frames=total_frames,
                duration_sec=duration_sec,
                format=format_ext
            )
        finally:
            cap.release()

    def extract_frames(self, video_path: str | Path, output_dir: str | Path) -> tuple[VideoMetadata, List[FrameInfo]]:
        """
        Extract frames from the video at the configured target FPS.

        Args:
            video_path: Path to the video file.
            output_dir: Directory to save extracted frames.

        Returns:
            A tuple containing the VideoMetadata and a list of FrameInfo objects.
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Extracting metadata from {video_path}")
        metadata = self.get_metadata(video_path)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        extracted_frames: List[FrameInfo] = []
        
        target_fps = self.config.target_fps
        max_frames = self.config.max_frames
        
        if target_fps <= 0 or target_fps > metadata.fps:
            logger.info(f"Target FPS ({target_fps}) invalid or higher than source FPS ({metadata.fps}). Using source FPS.")
            frame_interval = 1
        else:
            frame_interval = max(1, int(metadata.fps / target_fps))

        logger.info(f"Extracting frames at interval {frame_interval} (target {target_fps} FPS)")

        frame_idx = 0
        extracted_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    timestamp_sec = frame_idx / metadata.fps if metadata.fps > 0 else 0.0
                    
                    filename = f"frame_{frame_idx:06d}.{self.config.output_format}"
                    out_path = output_dir / filename
                    
                    cv2.imwrite(str(out_path), frame)
                    
                    extracted_frames.append(FrameInfo(
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                        file_path=out_path
                    ))
                    
                    extracted_count += 1
                    
                    if extracted_count % 50 == 0:
                        logger.info(f"Extracted {extracted_count} frames...")
                        
                    if max_frames > 0 and extracted_count >= max_frames:
                        logger.info(f"Reached maximum frame limit ({max_frames})")
                        break

                frame_idx += 1
                
        finally:
            cap.release()

        logger.info(f"Extraction complete. Total frames extracted: {extracted_count}")
        return metadata, extracted_frames
