
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import yaml
from pathlib import Path

@dataclass
class TransformConfig:
    name: str
    scale: float = 1.0
    rotation_euler_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    translation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

@dataclass
class ProjectorConfig:
    name: str
    display_index: int = 1
    resolution: List[int] = field(default_factory=lambda: [1920, 1080])
    position_room: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    look_at_room: List[float] = field(default_factory=lambda: [0.0, 0.0, -1.0])
    fov_deg: float = 60.0

@dataclass
class SurfaceConfig:
    name: str
    projector: str
    type: str = "flat"
    dst_quad_pixels: List[List[int]] = field(default_factory=list)

@dataclass
class ContentSourceConfig:
    name: str
    type: str
    file: Optional[str] = None
    rgb: Optional[str] = None
    depth: Optional[str] = None
    scene_asset: Optional[str] = None
    transform: Optional[str] = None
    frame_style: Optional[str] = None
    light_vignette: bool = False
    parallax_amount: float = 0.0
    camera_motion: Optional[str] = None
    style: Optional[str] = None
    intensity: float = 1.0
    texture: Optional[str] = None
    render_preset: str = "balanced"
    guest_camera_room: Optional[Dict[str, Any]] = None

@dataclass
class ContentLayerConfig:
    source: str
    blend: str = "normal"

@dataclass
class SceneConfig:
    at: float
    surface: str
    content_layers: List[ContentLayerConfig]

@dataclass
class ShowConfig:
    name: str
    loop: bool = False
    duration: Optional[float] = None
    scenes: List[SceneConfig] = field(default_factory=list)

@dataclass
class ProjectionConfig:
    transforms: Dict[str, TransformConfig] = field(default_factory=dict)
    projectors: Dict[str, ProjectorConfig] = field(default_factory=dict)
    surfaces: Dict[str, SurfaceConfig] = field(default_factory=dict)
    content_sources: Dict[str, ContentSourceConfig] = field(default_factory=dict)
    shows: Dict[str, ShowConfig] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> 'ProjectionConfig':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        # Parse Transforms
        for name, cfg in data.get('transforms', {}).items():
            config.transforms[name] = TransformConfig(name=name, **cfg)
            
        # Parse Projectors
        for name, cfg in data.get('projectors', {}).items():
            config.projectors[name] = ProjectorConfig(name=name, **cfg)
            
        # Parse Surfaces
        for name, cfg in data.get('surfaces', {}).items():
            config.surfaces[name] = SurfaceConfig(name=name, **cfg)
            
        # Parse Content Sources
        for name, cfg in data.get('content_sources', {}).items():
            config.content_sources[name] = ContentSourceConfig(name=name, **cfg)
            
        # Parse Shows
        for name, cfg in data.get('shows', {}).items():
            scenes = []
            for scene_data in cfg.get('scenes', []):
                layers = [ContentLayerConfig(**l) for l in scene_data.get('content_layers', [])]
                scenes.append(SceneConfig(
                    at=scene_data.get('at', 0),
                    surface=scene_data.get('surface'),
                    content_layers=layers
                ))
            
            config.shows[name] = ShowConfig(
                name=name,
                loop=cfg.get('loop', False),
                duration=cfg.get('duration'),
                scenes=scenes
            )
            
        return config

