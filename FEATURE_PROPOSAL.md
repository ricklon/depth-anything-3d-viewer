
# Feature Proposal: Immersive Projection & Room-Mapping for `da3d`

## 1. Summary

This proposal extends `da3d` from a 3D depth/mesh viewer into a lightweight **immersive projection toolkit**.

Core idea:

* Use `da3d`’s depth / 3D capabilities (meshes, Gaussian splats, etc.)
* Align a **virtual environment** (lobby, gallery, etc.) with a **measured real room**
* Use one or more projectors to **map the virtual scene onto real walls/floors**, with optional augmented effects and interactivity.

Flagship example modes:

* A **“Continental-style lobby experience”**: guests stand in a real room while `da3d` projects a lobby-like environment onto the room’s surfaces, making it feel like they are inside that scene.
* A **“Virtual Gallery Room”**: plain room becomes a curated exhibition space, where each wall becomes a framed “screen” for 2D images, depth-enhanced scenes, or volumetric art.

> All references to specific films are conceptual. Users are expected to provide their own assets or “inspired-by” scenes that they have the rights to use.

---

## 2. Motivation

`da3d` already supports:

* Depth estimation → 3D point clouds / meshes
* Viewing and interacting with those scenes
* Live sources (webcam, screen, etc.)

The natural next step is to let users:

* Take **cinematic or artistic 3D content**,
* Align it with **physical rooms**, and
* **Project** it onto real walls and floors so people feel “inside” or surrounded by that content.

This makes `da3d` useful for:

* Pop-up installations / immersive demos
* Maker/hacklab projection experiments
* Classroom / lab teaching of depth / volumetric concepts
* Turning arbitrary rooms into **virtual galleries** that present 2D + 2.5D + 3D content

---

## 3. Goals & Non-Goals

### Goals

* Provide a **projection-mapping framework** in `da3d`:

  * Projectors, surfaces, homography-based mapping.
* Support **alignment between virtual scenes and real rooms** via consistent coordinate systems.
* Implement example experiences:

  * A **“lobby experience”** (Continental-style) that demonstrates environment-to-room mapping.
  * A **“virtual gallery room”** that demonstrates multiple “frames” on walls showing different types of content.
* Keep everything **config-driven** (YAML) and scriptable via CLI.
* Make it easy to reuse the same machinery for other immersive ideas (clubs, corridors, abstract environments, etc.).

### Non-Goals

* Become a full show-control/media-server product (DMX, multi-room timelines, etc.).
* Ship copyrighted film or proprietary gallery assets.
* Replace professional projection-mapping tools. This is a **developer/maker toolkit**, not a turnkey commercial solution.

---

## 4. User Stories / Use Cases

### UC1 – Single-Projector Lobby Experience

> “I have a measured room and a 3D lobby scene. I want people to stand in the room and feel like they’re in that lobby.”

* User measures their space or scans it into a room mesh.
* They have a 3D lobby representation (mesh or Gaussian splat) with known scale.
* They align the lobby scene to the room’s coordinate system.
* They calibrate a single projector to the main wall.
* `da3d` renders the lobby from the guest’s viewpoint and projects it onto the wall; perspective lines up enough to feel “inside” the scene.

---

### UC2 – Multi-Surface Room (Wall + Floor Strip)

> “I want the main wall and a floor strip to both be part of the environment.”

* Same as UC1, but with multiple surfaces:

  * `wall_main` and `floor_strip`.
* Each surface has its own homography / quad in projector space.
* The lobby floor pattern and lighting extend onto the physical floor, making the whole room feel more unified.

---

### UC3 – Augmented Lobby Effects

> “I want the lobby to subtly animate and feel alive.”

* On top of the base environment:

  * Animated lighting (e.g., chandeliers brightening/dimming).
  * Silhouettes of people walking behind columns.
  * A patterned “carpet” projected on the floor strip.
* `da3d` composites base and overlay sources and plays a looping, atmospheric experience.

---

### UC4 – Optional Interactivity (Lobby or Other Scenes)

> “If someone walks onto the ‘carpet’, the path lights up and the lobby reacts.”

* A tracking source (depth camera, overhead camera, or external system) sends guest positions in room coordinates.
* A simple rule engine in `da3d` modifies overlays based on where people stand.
* The projection reacts: highlighted paths, intensified lighting, silhouettes pausing/turning, etc.

---

### UC5 – Turn a Room into a Virtual Gallery

> “I have a plain room and a bunch of digital artworks (2D, depth-enhanced, 3D). I want to turn the room into a virtual gallery, where each wall becomes a framed, lit artwork I can curate with `da3d`.”

**Scenario**

* User has:

  * A small room or studio.
  * A set of artworks:

    * 2D images (JPEG/PNG),
    * Depth-enhanced stills (RGB + depth),
    * 3D scenes or Gaussian splats.
  * One or more projectors.

**Goal**

* Make the room feel like a gallery:

  * Walls become “hanging surfaces.”
  * Each “frame” is a virtual screen with its own artwork.
  * Artwork can be static, subtly animated, or volumetric.

**How it uses the feature set**

1. **Projectors & Surfaces**

   * Define a projector (`main`) as usual.
   * Define multiple `surfaces` per wall to act like individual “frames”:

   ```yaml
   surfaces:
     wall_left_frame_1:
       projector: main
       type: flat
       dst_quad_pixels: [...]  # from calibration

     wall_left_frame_2:
       projector: main
       type: flat
       dst_quad_pixels: [...]

     wall_right_frame_1:
       projector: main
       type: flat
       dst_quad_pixels: [...]
   ```

2. **ContentSources as artworks**

   * Each artwork is a `content_source`:

   ```yaml
   content_sources:
     gallery_piece_1:
       type: image
       file: "assets/gallery/piece1.png"
       frame_style: "gold"
       light_vignette: true

     gallery_piece_2:
       type: depth_image
       rgb: "assets/gallery/piece2_rgb.png"
       depth: "assets/gallery/piece2_depth.exr"
       parallax_amount: 0.1
       frame_style: "black"

     gallery_piece_3:
       type: mesh_scene
       scene_asset: "assets/gallery/piece3_scene.splat"
       camera_motion: "slow_orbit"

     gallery_idle_light:
       type: lighting_overlay
       style: "gallery_spotlights"
       intensity: 0.2
   ```

   * `image`: static art with subtle vignette/spotlight.
   * `depth_image`: slight parallax for a 2.5D effect.
   * `mesh_scene` / `splat`: a deep “window” into a 3D scene.

3. **Show / Playlist as an Exhibition**

   ```yaml
   shows:
     virtual_gallery:
       loop: true
       scenes:
         - at: 0
           surface: wall_left_frame_1
           content_layers:
             - source: gallery_piece_1
               blend: normal
             - source: gallery_idle_light
               blend: additive

         - at: 0
           surface: wall_left_frame_2
           content_layers:
             - source: gallery_piece_2
               blend: normal
             - source: gallery_idle_light
               blend: additive

         - at: 0
           surface: wall_right_frame_1
           content_layers:
             - source: gallery_piece_3
               blend: normal
             - source: gallery_idle_light
               blend: additive
   ```

   * Optionally, time-based rotations:

   ```yaml
       scenes:
         - at: 0
           surface: wall_left_frame_1
           content_layers:
             - source: gallery_piece_1

         - at: 60
           surface: wall_left_frame_1
           content_layers:
             - source: gallery_piece_2

         - at: 120
           surface: wall_left_frame_1
           content_layers:
             - source: gallery_piece_3
   ```

4. **Optional Interactivity**

   * When a viewer stands in front of a frame:

     * Increase spotlight intensity.
     * Slowly reveal extra detail (zoom, fade to depth-enhanced version, etc.).
   * When no one is nearby:

     * Frames idle in a low-energy, dimmed state.

This use case shows the system is general-purpose and not tied to a single cinematic environment.

---

## 5. Core Concepts & Data Model

### 5.1 Coordinate Systems

* **RoomCoords**: coordinate system for the real room.

  * Origin at chosen floor corner.
  * +X along main wall, +Y up, +Z into the room.
* **SceneCoords**: coordinate system of a virtual scene (lobby, gallery piece, etc.).
* A transform maps scene → room:

  ```yaml
  transforms:
    lobby_to_room:
      scale: 0.8
      rotation_euler_deg: [0, 90, 0]
      translation: [2.0, 0.0, -1.5]
  ```

Multiple transforms can be defined (e.g. `lobby_to_room`, `gallery_piece3_to_frame` if needed).

---

### 5.2 Projector

Represents a physical projector output:

```yaml
projectors:
  main:
    display_index: 1         # OS display id
    resolution: [1920, 1080]
    position_room: [x, y, z]
    look_at_room: [cx, cy, cz]
    fov_deg: 60
```

---

### 5.3 Surface

A mappable physical region in a projector’s image (wall patch, floor strip, “frame” region).

```yaml
surfaces:
  wall_main:
    projector: main
    type: flat
    dst_quad_pixels:  # filled by calibration
      - [px0, py0]
      - [px1, py1]
      - [px2, py2]
      - [px3, py3]

  floor_strip:
    projector: main
    type: flat
    dst_quad_pixels: [...]
```

Later, we can add optional `physical_corners_room` to tie surfaces explicitly to room geometry.

---

### 5.4 ContentSource

Anything that can render a frame (or texture) per time `t`.

Examples:

* `lobby_scene`: renders a lobby-like scene from a defined guest camera in RoomCoords.
* `image`: 2D artwork with optional framing and lighting.
* `depth_image`: 2D RGB + depth for parallax effects.
* `mesh_scene`: volumetric piece (e.g., Gaussian splat).
* `lighting_overlay`: decorative/ambient lighting pass.
* `pattern_floor`: patterned carpet / gallery floor, etc.

Example:

```yaml
content_sources:
  lobby_base:
    type: lobby_scene
    scene_asset: "assets/lobby_scene.splat"
    transform: lobby_to_room
    guest_camera_room:
      position: [2.5, 1.6, -1.0]
      look_at:  [2.5, 1.6, 2.0]
      fov_deg: 70
    render_preset: balanced

  lobby_lighting:
    type: lighting_overlay
    style: "chandelier_pulse"
    intensity: 0.25

  lobby_carpet:
    type: pattern_floor
    texture: "assets/textures/continental_carpet_inspired.png"

  gallery_piece_1:
    type: image
    file: "assets/gallery/piece1.png"
    frame_style: "gold"
    light_vignette: true
```

---

### 5.5 Show / Timeline

Defines how content sources map to surfaces over time:

```yaml
shows:
  continental_lobby:
    loop: true
    duration: 300
    scenes:
      - at: 0
        surface: wall_main
        content_layers:
          - source: lobby_base
            blend: normal
          - source: lobby_lighting
            blend: additive

      - at: 0
        surface: floor_strip
        content_layers:
          - source: lobby_carpet
            blend: normal

  virtual_gallery:
    loop: true
    scenes:
      - at: 0
        surface: wall_left_frame_1
        content_layers:
          - source: gallery_piece_1
            blend: normal
```

---

## 6. CLI Extensions

Extend `da3d` CLI with projection-related subcommands.

### 6.1 Calibration

```bash
da3d projector-calibrate \
  --config config/projection.yaml \
  --projector main
```

* Fullscreen window on selected display.
* For each surface:

  * Shows a quad with draggable corners.
  * User aligns to taped corners on wall/floor.
* Writes `dst_quad_pixels` into config.

### 6.2 Projection Preview

```bash
da3d projector-preview \
  --config config/projection.yaml \
  --show continental_lobby
```

* Renders a specified show onto projector(s).
* Useful for verifying calibration and simple test content.

### 6.3 Run Show

```bash
da3d run-show \
  --config config/projection.yaml \
  --show virtual_gallery
```

* Opens windows on projector displays.
* Loads `content_sources` and `shows`.
* Frame loop:

  * Pick current scene/time.
  * Render `ContentSource` frames/textures.
  * Warp and composite them onto each surface via homographies.

### 6.4 Optional Live Mode

```bash
da3d projector-live \
  --config config/projection.yaml \
  --source some_live_source
```

* Same as `run-show`, but one or more content sources are live (webcam depth, screen3d, etc.).

---

## 7. Rendering & Mapping Pipeline

### 7.1 Content Rendering

Each `ContentSource` implements:

```python
class ContentSource:
    def render_frame(self, t: float) -> Frame:
        ...
```

* `Frame` might be:

  * A numpy array (CPU path), or
  * An OpenGL texture handle (GPU path).

Examples:

* `lobby_scene`:

  * Load lobby scene (mesh/splat).
  * Apply `lobby_to_room` transform.
  * Position camera in RoomCoords.
  * Render to offscreen target.

* `image`:

  * Load from disk.
  * Maybe apply simple shader for vignetting/spotlighting.

* `depth_image`:

  * Use depth + camera motion to produce gentle parallax.

* `mesh_scene`:

  * Render volumetric scene (Gaussian splat) from a small camera orbit.

---

### 7.2 Mapping to Projectors

#### Option A – CPU Prototype (OpenCV)

1. `Frame` is `(H, W, 3)` numpy array.
2. For each surface:

   * Use its `dst_quad_pixels` and a normalized source quad to compute homography `H`.
   * `cv2.warpPerspective(frame, H, (proj_w, proj_h))`.
3. Composite with alpha/`np.maximum`.

Good for initial implementation and debugging.

#### Option B – GPU (Recommended for real use)

* Use `da3d`’s OpenGL context.
* For each content source:

  * Render into an offscreen FBO texture.
* For each surface:

  * Draw a quad covering the relevant region of the projector framebuffer.
  * Fragment shader uses inverse homography `H⁻¹` to sample content texture.

---

## 8. Example Config: `config/projection_example.yaml`

```yaml
transforms:
  lobby_to_room:
    scale: 0.85
    rotation_euler_deg: [0, 90, 0]
    translation: [2.0, 0.0, -1.5]

projectors:
  main:
    display_index: 1
    resolution: [1920, 1080]
    position_room: [0.0, 2.2, -3.5]
    look_at_room: [2.5, 1.5, 0.0]
    fov_deg: 60

surfaces:
  wall_main:
    projector: main
    type: flat
    dst_quad_pixels:  # filled by projector-calibrate
      - [100, 100]
      - [1820, 100]
      - [1820, 980]
      - [100, 980]

  floor_strip:
    projector: main
    type: flat
    dst_quad_pixels:
      - [400, 900]
      - [1520, 900]
      - [1520, 1080]
      - [400, 1080]

  wall_left_frame_1:
    projector: main
    type: flat
    dst_quad_pixels:
      - [120, 200]
      - [520, 200]
      - [520, 600]
      - [120, 600]

content_sources:
  lobby_base:
    type: lobby_scene
    scene_asset: "assets/lobby_scene.splat"
    transform: lobby_to_room
    guest_camera_room:
      position: [2.5, 1.6, -1.0]
      look_at:  [2.5, 1.6, 2.0]
      fov_deg: 70
    render_preset: balanced

  lobby_lighting:
    type: lighting_overlay
    style: "chandelier_pulse"
    intensity: 0.25

  lobby_carpet:
    type: pattern_floor
    texture: "assets/textures/continental_carpet_inspired.png"

  gallery_piece_1:
    type: image
    file: "assets/gallery/piece1.png"
    frame_style: "gold"
    light_vignette: true

shows:
  continental_lobby:
    loop: true
    duration: 300
    scenes:
      - at: 0
        surface: wall_main
        content_layers:
          - source: lobby_base
            blend: normal
          - source: lobby_lighting
            blend: additive

      - at: 0
        surface: floor_strip
        content_layers:
          - source: lobby_carpet
            blend: normal

  virtual_gallery:
    loop: true
    scenes:
      - at: 0
        surface: wall_left_frame_1
        content_layers:
          - source: gallery_piece_1
            blend: normal
```

---

## 9. Implementation Plan (Milestones)

1. **Config & Concepts**

   * Define YAML schema for `projectors`, `surfaces`, `content_sources`, `shows`, `transforms`.
   * Implement parsing in `da3d.projection.config`.

2. **Calibration UI**

   * Implement `projector-calibrate`:

     * Fullscreen on a chosen display.
     * Draggable quad for each surface.
     * Writes `dst_quad_pixels`.

3. **CPU Prototype**

   * Implement `run-show` with CPU path (OpenCV `warpPerspective`).
   * Support:

     * One projector,
     * One or more surfaces,
     * Simple `image` / test-pattern content source.

4. **Lobby Scene Integration**

   * Add `LobbySceneSource` (type `lobby_scene`).
   * Load a test 3D scene (even a dummy cube room).
   * Apply `lobby_to_room` transform and guest camera.
   * Get end-to-end projection on `wall_main`.

5. **Multi-Surface & Layers**

   * Extend compositor to handle multiple surfaces and multiple `content_layers` per surface.

6. **GPU Path**

   * Implement GLSL-based homography in existing GL renderer.
   * Allow switching between CPU/GPU mapping for debugging.

7. **Gallery Sources**

   * Implement `image`, `depth_image`, and `mesh_scene` content sources.
   * Implement simple “frame” + spotlight shader for gallery pieces.

8. **Interactivity (Optional)**

   * Define an interface for external tracking input (JSON/OSC/IPC).
   * Expose simple parameter hooks (e.g., `viewer_position` passed into content sources).

9. **Examples & Docs**

   * Provide:

     * A generic “lobby-like” example (no copyrighted assets).
  * Maybe apply simple shader for vignetting/spotlighting.

* `depth_image`:

  * Use depth + camera motion to produce gentle parallax.

* `mesh_scene`:

  * Render volumetric scene (Gaussian splat) from a small camera orbit.

---

### 7.2 Mapping to Projectors

#### Option A – CPU Prototype (OpenCV)

1. `Frame` is `(H, W, 3)` numpy array.
2. For each surface:

   * Use its `dst_quad_pixels` and a normalized source quad to compute homography `H`.
   * `cv2.warpPerspective(frame, H, (proj_w, proj_h))`.
3. Composite with alpha/`np.maximum`.

Good for initial implementation and debugging.

#### Option B – GPU (Recommended for real use)

* Use `da3d`’s OpenGL context.
* For each content source:

  * Render into an offscreen FBO texture.
* For each surface:

  * Draw a quad covering the relevant region of the projector framebuffer.
  * Fragment shader uses inverse homography `H⁻¹` to sample content texture.

---

## 8. Example Config: `config/projection_example.yaml`

```yaml
transforms:
  lobby_to_room:
    scale: 0.85
    rotation_euler_deg: [0, 90, 0]
    translation: [2.0, 0.0, -1.5]

projectors:
  main:
    display_index: 1
    resolution: [1920, 1080]
    position_room: [0.0, 2.2, -3.5]
    look_at_room: [2.5, 1.5, 0.0]
    fov_deg: 60

surfaces:
  wall_main:
    projector: main
    type: flat
    dst_quad_pixels:  # filled by projector-calibrate
      - [100, 100]
      - [1820, 100]
      - [1820, 980]
      - [100, 980]

  floor_strip:
    projector: main
    type: flat
    dst_quad_pixels:
      - [400, 900]
      - [1520, 900]
      - [1520, 1080]
      - [400, 1080]

  wall_left_frame_1:
    projector: main
    type: flat
    dst_quad_pixels:
      - [120, 200]
      - [520, 200]
      - [520, 600]
      - [120, 600]

content_sources:
  lobby_base:
    type: lobby_scene
    scene_asset: "assets/lobby_scene.splat"
    transform: lobby_to_room
    guest_camera_room:
      position: [2.5, 1.6, -1.0]
      look_at:  [2.5, 1.6, 2.0]
      fov_deg: 70
    render_preset: balanced

  lobby_lighting:
    type: lighting_overlay
    style: "chandelier_pulse"
    intensity: 0.25

  lobby_carpet:
    type: pattern_floor
    texture: "assets/textures/continental_carpet_inspired.png"

  gallery_piece_1:
    type: image
    file: "assets/gallery/piece1.png"
    frame_style: "gold"
    light_vignette: true

shows:
  continental_lobby:
    loop: true
    duration: 300
    scenes:
      - at: 0
        surface: wall_main
        content_layers:
          - source: lobby_base
            blend: normal
          - source: lobby_lighting
            blend: additive

      - at: 0
        surface: floor_strip
        content_layers:
          - source: lobby_carpet
            blend: normal

  virtual_gallery:
    loop: true
    scenes:
      - at: 0
        surface: wall_left_frame_1
        content_layers:
          - source: gallery_piece_1
            blend: normal
```

---

## 9. Implementation Plan (Milestones)

1. **Config & Concepts**

   * Define YAML schema for `projectors`, `surfaces`, `content_sources`, `shows`, `transforms`.
   * Implement parsing in `da3d.projection.config`.

2. **Calibration UI**

   * Implement `projector-calibrate`:

     * Fullscreen on a chosen display.
     * Draggable quad for each surface.
     * Writes `dst_quad_pixels`.

3. **CPU Prototype**

   * Implement `run-show` with CPU path (OpenCV `warpPerspective`).
   * Support:

     * One projector,
     * One or more surfaces,
     * Simple `image` / test-pattern content source.

4. **Lobby Scene Integration**

   * Add `LobbySceneSource` (type `lobby_scene`).
   * Load a test 3D scene (even a dummy cube room).
   * Apply `lobby_to_room` transform and guest camera.
   * Get end-to-end projection on `wall_main`.

5. **Multi-Surface & Layers**

   * Extend compositor to handle multiple surfaces and multiple `content_layers` per surface.

6. **GPU Path**

   * Implement GLSL-based homography in existing GL renderer.
   * Allow switching between CPU/GPU mapping for debugging.

7. **Gallery Sources**

   * Implement `image`, `depth_image`, and `mesh_scene` content sources.
   * Implement simple “frame” + spotlight shader for gallery pieces.

8. **Interactivity (Optional)**

   * Define an interface for external tracking input (JSON/OSC/IPC).
   * Expose simple parameter hooks (e.g., `viewer_position` passed into content sources).

9. **Examples & Docs**

   * Provide:

     * A generic “lobby-like” example (no copyrighted assets).
     * A “virtual gallery” example with placeholder art.
   * Document workflows:

     * `docs/projection_workflow.md`
     * `docs/projection_lobby_example.md`
     * `docs/projection_gallery_example.md`
