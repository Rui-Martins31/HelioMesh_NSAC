import sys
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_ESCAPE
from OpenGL.GL import *
from OpenGL.GLU import *
import time
import dynamics as dyn


# ---------- simple draw helpers ----------
def draw_grid(size=10, step=1):
    glDisable(GL_LIGHTING)
    glLineWidth(1.0)
    glColor3f(0.28, 0.28, 0.3)
    glBegin(GL_LINES)
    for i in range(-size, size + 1, step):
        glVertex3f(i, 0, -size); glVertex3f(i, 0, size)
        glVertex3f(-size, 0, i);  glVertex3f(size, 0, i)
    glEnd()
    glEnable(GL_LIGHTING)

def draw_ground(size=10):
    glDisable(GL_LIGHTING)
    glColor3f(0.12, 0.12, 0.14)
    glBegin(GL_QUADS)
    glVertex3f(-size, 0, -size); glVertex3f(size, 0, -size)
    glVertex3f(size, 0, size);   glVertex3f(-size, 0, size)
    glEnd()
    glEnable(GL_LIGHTING)

def draw_cube(center=(0,0,0), half=0.5):
    cx, cy, cz = center
    v = [
        (cx-half, cy-half, cz-half), (cx+half, cy-half, cz-half),
        (cx+half, cy+half, cz-half), (cx-half, cy+half, cz-half),
        (cx-half, cy-half, cz+half), (cx+half, cy-half, cz+half),
        (cx+half, cy+half, cz+half), (cx-half, cy+half, cz+half),
    ]
    faces = [
        ((0,1,2,3), (0,0,-1)),  # -Z
        ((4,5,6,7), (0,0, 1)),  # +Z
        ((0,4,7,3), (-1,0,0)),  # -X
        ((1,5,6,2), ( 1,0,0)),  # +X
        ((3,2,6,7), (0, 1,0)),  # +Y (top)
        ((0,1,5,4), (0,-1,0)),  # -Y (bottom)
    ]
    glBegin(GL_QUADS)
    for idxs, n in faces:
        glNormal3f(*n)
        glColor3f(0.70 - 0.15*abs(n[0]), 0.72 + 0.18*(n[1] > 0), 0.78 + 0.12*abs(n[2]))
        for i in idxs: glVertex3f(*v[i])
    glEnd()

def draw_square_from_vertex(anchor, dir_x, dir_z, edge=0.5, y_normal_up=True):
    ax, ay, az = anchor

    # base corners (from the shared vertex)
    p0 = (ax,                ay,                az)
    p1 = (ax + dir_x*edge,   ay,                az)
    p2 = (ax + dir_x*edge,   ay,                az + dir_z*edge)
    p3 = (ax,                ay,                az + dir_z*edge)

    # Ensure consistent winding: for a square on y=const,
    # (p1-p0) x (p2-p0) = (0, -dir_x*dir_z*edge^2, 0)
    # So if dir_x*dir_z > 0, the current order gives -Y; flip it.
    same_sign = (dir_x * dir_z) > 0

    if y_normal_up:
        order = (p0, p3, p2, p1) if same_sign else (p0, p1, p2, p3)
        nx, ny, nz = (0, 1, 0)
    else:
        # For -Y, reverse the CCW rule
        order = (p0, p1, p2, p3) if same_sign else (p0, p3, p2, p1)
        nx, ny, nz = (0, -1, 0)

    glBegin(GL_QUADS)
    glNormal3f(nx, ny, nz)
    glColor3f(0.85, 0.86, 0.90)
    for p in order:
        glVertex3f(*p)
    glEnd()

    # outline (lines don't use face culling)
    glDisable(GL_LIGHTING)
    glLineWidth(1.5)
    glColor3f(0.93, 0.93, 0.97)
    glBegin(GL_LINE_LOOP)
    for p in order:
        glVertex3f(*p)
    glEnd()
    glEnable(GL_LIGHTING)


# ---------- orientation helper ----------
def apply_orientation_face_X():
    """
    Rotate the whole structure so its 'top' (+Y in object space)
    faces the world +X axis. Equivalent to a -90° rotation around Z.
    """
    glRotatef(-90.0, 0.0, 0.0, 1.0)

def apply_pose(state: dyn.State):
    """Translate + rotate model by current pose."""
    # translate
    glTranslatef(*state.pos.astype(float))
    # rotate via quaternion
    R = dyn.quat_to_mat3(state.q).astype(np.float32)
    M = np.eye(4, dtype=np.float32)
    M[:3,:3] = R
    glMultMatrixf(M.T)  # OpenGL expects column-major

# ---------- app ----------
def main():
    pygame.init()
    w, h = 1100, 700
    pygame.display.set_mode((w, h), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("HelioMesh")

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE); glCullFace(GL_BACK)
    glEnable(GL_LIGHTING);  glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (5.0, 8.0, 6.0, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  (0.9, 0.9, 0.9, 1.0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (0.6, 0.6, 0.6, 1.0))
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0.18, 0.18, 0.18, 1.0))
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 16.0)
    glClearColor(0.05, 0.06, 0.08, 1.0)

    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluPerspective(60.0, w/float(h), 0.1, 200.0)
    glMatrixMode(GL_MODELVIEW)

    eye  = np.array([20.0, 5.0, 12.5], dtype=float)
    look = np.array([0.0, 0.7, 0.0], dtype=float)
    up   = np.array([0.0, 1.0, 0.0], dtype=float)

    central_width = 1
    panel_width   = 10

    state = dyn.State()
    # Example demo motion (remove once you plug your dynamics)
    # state.vel = np.array([0.0, 0.0, 0.0])   # drift
    # state.w   = np.array([0.0, 0.3, 0.0])   # spin around Y
    last = time.perf_counter()

    clock = pygame.time.Clock()
    running = True
    while running:
        # Events
        for ev in pygame.event.get():
            if ev.type == QUIT:
                running = False
            elif ev.type == KEYDOWN and ev.key == K_ESCAPE:
                running = False

        # Time step
        now = time.perf_counter()
        dt = min(0.05, now - last)   # clamp to keep stable if the window stalls
        last = now

        # Update dynamics
        runtime_ms = pygame.time.get_ticks()
        dyn.step(state, dt, runtime_ms)

        # Render
        glViewport(0, 0, w, h)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        gluLookAt(*(eye.tolist() + look.tolist() + up.tolist()))

        # World guides
        draw_ground(12)
        draw_grid(12, 1)

        # Draw the structure with its world pose
        glPushMatrix()
        apply_pose(state)                 # translate + quaternion rotate
        glRotatef(-90.0, 0.0, 0.0, 1.0)   # keep object “facing +X” mapping

        # Object-space geometry
        draw_cube(center=(0, 0, 0), half=central_width)

        top_y = +central_width
        vertices = [
            (+central_width, top_y, +central_width),
            (-central_width, top_y, +central_width),
            (+central_width, top_y, -central_width),
            (-central_width, top_y, -central_width),
        ]
        for vx, vy, vz in vertices:
            dir_x = 1 if vx > 0 else -1
            dir_z = 1 if vz > 0 else -1
            draw_square_from_vertex(
                (vx, vy, vz), 
                dir_x, 
                dir_z,
                edge=panel_width, 
                y_normal_up=True
            )

        glPopMatrix()

        pygame.display.flip()
        clock.tick(120)

    pygame.quit(); sys.exit()

if __name__ == "__main__":
    main()
