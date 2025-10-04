import numpy as np

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v if n < eps else v / n

def quat_mul(q, r):
    w1,x1,y1,z1 = q; w2,x2,y2,z2 = r
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def quat_to_mat3(q):
    w,x,y,z = q
    xx,yy,zz = x*x,y*y,z*z
    xy,xz,yz = x*y,x*z,y*z
    wx,wy,wz = w*x,w*y,w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=float)

class State:
    """State of the body."""
    def __init__(self):
        self.pos = np.zeros(3)            # world position
        self.vel = np.zeros(3)            # world linear velocity
        self.q   = np.array([1,0,0,0.0])  # orientation quaternion (w,x,y,z)
        self.w   = np.zeros(3)            # angular velocity (rad/s)

def step(state: State, dt: float, runtime_duration: float):
    """Updates the simulation."""
    # Position
    x_vel: int = np.random.uniform(-1, 1)   # Random for now
    y_vel: int = np.random.uniform(-1, 1)   # Random for now
    z_vel: int = np.random.uniform(-1, 1)   # Random for now
    state.vel  = np.array([x_vel, y_vel, z_vel]) 
    state.pos  += state.vel * dt

    # Angular velocity
    x_w: int   = 0.0
    y_w: int   = dt * 10 # np.sin(runtime_duration)
    z_w: int   = 0.0
    state.w    = np.array([x_w, y_w, z_w])

    # Update quaternion
    dq = 0.5 * quat_mul(np.array([0.0, *state.w]), state.q)
    state.q = normalize(state.q + dq * dt)
    return state
