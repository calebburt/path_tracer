import argparse
import subprocess
import sys
from pathlib import Path

import glfw
from OpenGL.GL import *
import numpy as np
from PIL import Image

# GPU layout must match the GLSL std430 packing of:
#   struct Material {
#       vec3 emissive; vec3 albedo; uint reflects;
#       float roughness; float metallic;
#   };
#   struct Triangle { vec3 v0; vec3 v1; vec3 v2; Material material; };
# Material occupies 48 bytes (vec3-alignment rounding), Triangle is 96 bytes.
triangle_dtype = np.dtype([
    ("v0",       np.float32, 3),
    ("_pad0",    np.float32),

    ("v1",       np.float32, 3),
    ("_pad1",    np.float32),

    ("v2",       np.float32, 3),
    ("_pad2",    np.float32),

    ("emissive", np.float32, 3),
    ("_pad3",    np.float32),

    ("albedo",   np.float32, 3),
    ("reflects", np.uint32),

    ("roughness", np.float32),
    ("metallic",  np.float32),
    ("_pad4",     np.float32, 2),  # round Material to 48B (vec3-aligned)
])
assert triangle_dtype.itemsize == 96


MaterialTuple = tuple[
    tuple[float, float, float],  # emissive (Ke)
    tuple[float, float, float],  # albedo   (Kd)
    float,                       # roughness (Pr)
    float,                       # metallic  (Pm)
]


def _parse_mtl(path: Path) -> dict[str, MaterialTuple]:
    """Parse a Wavefront .mtl file. Returns name -> (emissive, albedo, roughness, metallic)."""
    materials: dict[str, MaterialTuple] = {}
    if not path.exists():
        return materials

    name: str | None = None
    kd = (1.0, 1.0, 1.0)
    ke = (0.0, 0.0, 0.0)
    pr = 1.0   # default fully rough
    pm = 0.0   # default dielectric

    def flush() -> None:
        if name is not None:
            materials[name] = (ke, kd, pr, pm)

    with open(path) as f:
        for raw in f:
            parts = raw.split()
            if not parts or parts[0].startswith("#"):
                continue
            tag = parts[0]
            if tag == "newmtl":
                flush()
                name = parts[1]
                kd = (1.0, 1.0, 1.0)
                ke = (0.0, 0.0, 0.0)
                pr = 1.0
                pm = 0.0
            elif tag == "Kd" and len(parts) >= 4:
                kd = (float(parts[1]), float(parts[2]), float(parts[3]))
            elif tag == "Ke" and len(parts) >= 4:
                ke = (float(parts[1]), float(parts[2]), float(parts[3]))
            elif tag == "Pr" and len(parts) >= 2:
                pr = float(parts[1])
            elif tag == "Pm" and len(parts) >= 2:
                pm = float(parts[1])
    flush()
    return materials


def load_triangles_from_obj(path: str) -> np.ndarray:
    """Parse a Wavefront .obj (with optional .mtl) and pack triangles for the GPU."""
    obj_path = Path(path)
    obj_dir = obj_path.parent

    vertices: list[tuple[float, float, float]] = []
    materials: dict[str, MaterialTuple] = {}
    faces: list[tuple[int, int, int, str | None]] = []
    current_mat: str | None = None

    with open(obj_path) as f:
        for raw in f:
            parts = raw.split()
            if not parts or parts[0].startswith("#"):
                continue
            tag = parts[0]
            if tag == "v" and len(parts) >= 4:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif tag == "mtllib" and len(parts) >= 2:
                materials.update(_parse_mtl(obj_dir / parts[1]))
            elif tag == "usemtl" and len(parts) >= 2:
                current_mat = parts[1]
            elif tag == "f" and len(parts) >= 4:
                # face vertex tokens: "v", "v/vt", "v//vn" or "v/vt/vn"; 1-based, negative = from end
                idx: list[int] = []
                for token in parts[1:]:
                    v = int(token.split("/")[0])
                    idx.append(v - 1 if v > 0 else len(vertices) + v)
                # triangulate as a fan
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1], current_mat))

    verts = np.asarray(vertices, dtype=np.float32)
    out = np.zeros(len(faces), dtype=triangle_dtype)

    default_mat: MaterialTuple = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1.0, 0.0)
    for i, (a, b, c, mat_name) in enumerate(faces):
        emissive, albedo, roughness, metallic = (
            materials.get(mat_name, default_mat) if mat_name else default_mat
        )
        out[i]["v0"] = verts[a]
        out[i]["v1"] = verts[b]
        out[i]["v2"] = verts[c]
        out[i]["emissive"] = emissive
        out[i]["albedo"] = albedo
        out[i]["reflects"] = 1 if any(ch != 0.0 for ch in albedo) else 0
        out[i]["roughness"] = roughness
        out[i]["metallic"] = metallic

    return np.ascontiguousarray(out)

# ============================================================
# Shaders
# ============================================================

VERTEX_SHADER_SRC = """
#version 330 core
layout(location = 0) in vec2 position;
out vec2 fragCoord;
void main() {
    fragCoord = position;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

FRAGMENT_SHADER_SRC = open("shader.glsl", "r").read()

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader).decode())
    return shader

def create_shader_program():
    vs = compile_shader(VERTEX_SHADER_SRC, GL_VERTEX_SHADER)
    fs = compile_shader(FRAGMENT_SHADER_SRC, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(program).decode())
    glDeleteShader(vs)
    glDeleteShader(fs)
    return program

def framebuffer_size_callback(window, width, height):
    glViewport(0, 0, width, height)


def init_glfw_window(width: int, height: int, visible: bool) -> "glfw._GLFWwindow":
    if not glfw.init():
        print("Failed to initialize GLFW")
        sys.exit(1)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)
    window = glfw.create_window(width, height, "Path Tracer", None, None)
    if not window:
        glfw.terminate()
        sys.exit(1)
    glfw.make_context_current(window)
    return window


def setup_scene(obj_path: str):
    """Compile shader, upload triangles, return (program, num_triangles, uniform_locs)."""
    triangles = np.ascontiguousarray(load_triangles_from_obj(obj_path))
    program = create_shader_program()

    vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    ], dtype=np.float32)

    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, triangles.nbytes, triangles, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo)

    locs = {
        "iTime":        glGetUniformLocation(program, "iTime"),
        "iApertureSize":glGetUniformLocation(program, "iApertureSize"),
        "iResolution":  glGetUniformLocation(program, "iResolution"),
        "numTriangles": glGetUniformLocation(program, "numTriangles"),
        "iSamples":     glGetUniformLocation(program, "iSamples"),
        "iBackground":  glGetUniformLocation(program, "iBackground"),
    }
    return program, len(triangles), locs


# ============================================================
# Live preview
# ============================================================

def run_preview(
    obj_path: str,
    aperture_size: float,
    samples: int,
    width: int,
    height: int,
    background: tuple[float, float, float],
) -> None:
    window = init_glfw_window(width, height, visible=True)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)

    program, num_tris, locs = setup_scene(obj_path)

    start = glfw.get_time()
    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(program)

        w, h = glfw.get_framebuffer_size(window)
        glUniform1f(locs["iTime"], glfw.get_time() - start)
        glUniform2f(locs["iResolution"], float(w), float(h))
        glUniform1f(locs["iApertureSize"], aperture_size)
        glUniform1i(locs["numTriangles"], num_tris)
        glUniform1i(locs["iSamples"], samples)
        glUniform3f(locs["iBackground"], *background)

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


# ============================================================
# Offline render → ffmpeg
# ============================================================

def run_render(
    obj_path: str,
    aperture_size: float,
    out_path: str,
    width: int,
    height: int,
    fps: int,
    duration: float,
    samples: int,
    tile: int,
    background: tuple[float, float, float],
) -> None:
    """Render frames to an offscreen FBO in tiles (sidesteps GPU watchdog) and pipe to ffmpeg."""
    window = init_glfw_window(width, height, visible=False)
    program, num_tris, locs = setup_scene(obj_path)

    # Offscreen color target
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    color_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, color_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex, 0)
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("FBO incomplete")

    glPixelStorei(GL_PACK_ALIGNMENT, 1)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        "-vf", "vflip",                     # OpenGL is bottom-up
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        out_path,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None

    n_frames = int(round(duration * fps))
    frame_buf = np.empty((height, width, 3), dtype=np.uint8)

    glUseProgram(program)
    glUniform2f(locs["iResolution"], float(width), float(height))
    glUniform1f(locs["iApertureSize"], aperture_size)
    glUniform1i(locs["numTriangles"], num_tris)
    glUniform1i(locs["iSamples"], samples)
    glUniform3f(locs["iBackground"], *background)

    try:
        for f in range(n_frames):
            t = f / fps
            glUniform1f(locs["iTime"], t)

            for ty in range(0, height, tile):
                for tx in range(0, width, tile):
                    tw = min(tile, width - tx)
                    th = min(tile, height - ty)
                    glViewport(tx, ty, tw, th)
                    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
                    glFinish()  # complete each tile separately so the watchdog never sees a long submit

            glReadBuffer(GL_COLOR_ATTACHMENT0)
            glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, frame_buf)
            proc.stdin.write(frame_buf.tobytes())
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
        glfw.terminate()


def run_photo(
    obj_path: str,
    aperture_size: float,
    out_path: str,
    width: int,
    height: int,
    time: float,
    samples: int,
    tile: int,
    background: tuple[float, float, float],
) -> None:
    """Render a single frame at given time and save to an image file."""
    window = init_glfw_window(width, height, visible=False)
    program, num_tris, locs = setup_scene(obj_path)

    # Offscreen color target
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    color_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, color_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex, 0)
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("FBO incomplete")

    glPixelStorei(GL_PACK_ALIGNMENT, 1)

    frame_buf = np.empty((height, width, 3), dtype=np.uint8)

    glUseProgram(program)
    glUniform2f(locs["iResolution"], float(width), float(height))
    glUniform1f(locs["iApertureSize"], aperture_size)
    glUniform1i(locs["numTriangles"], num_tris)
    glUniform1i(locs["iSamples"], samples)
    glUniform3f(locs["iBackground"], *background)

    try:
        glUniform1f(locs["iTime"], time)
        for ty in range(0, height, tile):
            for tx in range(0, width, tile):
                tw = min(tile, width - tx)
                th = min(tile, height - ty)
                glViewport(tx, ty, tw, th)
                glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
                glFinish()
                print("Done tile", tx, ty)

        glReadBuffer(GL_COLOR_ATTACHMENT0)
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, frame_buf)
        # OpenGL is bottom-up; flip to top-down for image files
        img = np.flipud(frame_buf)
        Image.fromarray(img, mode="RGB").save(out_path)
    finally:
        glfw.terminate()


# ============================================================
# Entry point
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obj", default="cornell_box.obj")
    p.add_argument("--aperture", type=float, default=1)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=600)
    p.add_argument("--render", metavar="OUT.mp4", help="render to a video file instead of showing a window")
    p.add_argument("--photo", metavar="OUT.png", help="render a single frame and save to an image file")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--duration", type=float, default=5.0, help="seconds (render mode)")
    p.add_argument("--time", type=float, default=0.0, help="time value for photo mode")
    p.add_argument("--tile", type=int, default=128, help="tile size for offline render (smaller = safer for watchdog)")
    p.add_argument("--background", type=float, nargs=3, metavar=("R", "G", "B"),
                   default=[0.5, 0.7, 1.0],
                   help="sky/background color in linear RGB (default: 0.5 0.7 1.0)")
    args = p.parse_args()

    background = (args.background[0], args.background[1], args.background[2])

    if args.photo:
        run_photo(args.obj, args.aperture, args.photo, args.width, args.height,
                  args.time, args.samples, args.tile, background)
    elif args.render:
        run_render(args.obj, args.aperture, args.render, args.width, args.height,
                   args.fps, args.duration, args.samples, args.tile, background)
    else:
        run_preview(args.obj, args.aperture, args.samples, args.width, args.height, background)


if __name__ == "__main__":
    main()
