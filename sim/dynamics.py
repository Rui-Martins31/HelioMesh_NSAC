import numpy as np

# ---------- quaternions ----------
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

# ---------- dynamics ----------
class State:
    """State of the body."""
    def __init__(self):
        self.pos = np.zeros(3)            # world position (m)
        self.vel = np.zeros(3)            # world linear velocity (m/s)
        self.q   = np.array([1,0,0,0.0])  # orientation quaternion (w,x,y,z)
        self.w   = np.zeros(3)            # angular velocity (rad/s)

        self.mass = 100                   # mass (Kg) 
        self.size = 21                    # bounding box side length (m)

def step(state: State, dt: float, runtime_duration_ms: float):
    """Updates the simulation."""
    # Correct previous disturbance
    pid_control()

    # Forces
    F_ext, T_ext = external_forces(runtime_duration_ms/1000)
    # print(f"{F_ext}")

    # Position
    x_vel: int = (F_ext[0]/state.mass) * dt # integrate
    y_vel: int = (F_ext[1]/state.mass) * dt
    z_vel: int = (F_ext[2]/state.mass) * dt
    state.vel  = np.array([x_vel, y_vel, z_vel]) 
    state.pos  += state.vel * dt

    # Angular velocity
    x_w: int   = T_ext[0]
    y_w: int   = T_ext[1]
    z_w: int   = T_ext[2]
    state.w    = np.array([x_w, y_w, z_w])

    # Update quaternion
    dq      = 0.5 * quat_mul(np.array([0.0, *state.w]), state.q)
    state.q = normalize(state.q + dq * dt)
    return state

def external_forces(runtime_duration: float):
    ## TO DO - Add a bounding box that helps to calculate torque values

    # Simulate space wind or radiation with random force primarily in the y-axis
    # Random force mainly in y
    x_force = np.random.uniform(-1.0, 1.0)   # Small perturbation
    y_force = np.random.uniform(-10.0, 10.0)
    z_force = np.random.uniform(-1.0, 1.0)   # Small perturbation
    F_ext   = np.array([x_force, y_force, z_force])

    # Rnadom torque
    T_ext   = np.random.uniform(-1.0, 1.0, 3)

    return F_ext, T_ext

def pid_control():
    pass
