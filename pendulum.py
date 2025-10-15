from pydrake.all import *
import matplotlib.pyplot as plt
import time
from visualization import *


def dare(A, B, Q, R, tol=1e-10, max_iter=10_000):
    P = Q.copy()
    for _ in range(max_iter):
        BtP = B.T @ P
        S = R + BtP @ B
        K = np.linalg.solve(S, BtP @ A)
        P_next = A.T @ P @ A - A.T @ P @ B @ K + Q
        if np.linalg.norm(P_next - P, ord='fro') <= tol * max(1.0, np.linalg.norm(P, ord='fro')):
            P = P_next
            break
        P = P_next
    else:
        print("[warn] DARE iteration hit max_iter without converging.")

    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return P, K

def dynamics(x, u, params):
    if x.dtype == object or u.dtype == object:
        plant : MultibodyPlant = params["plant_ad"]
        plant_context : Context = params["plant_ad_context"]
    else:
        plant : MultibodyPlant = params["plant"]
        plant_context : Context = params["plant_context"]
    plant.SetPositionsAndVelocities(plant_context, x)
    plant.get_actuation_input_port().FixValue(plant_context, u)

    return np.concatenate([plant.GetPositionsAndVelocities(plant_context)[2:], plant.get_generalized_acceleration_output_port().Eval(plant_context)])
    
def rk4(x, u, dynamics, dt):
    k1 = dynamics(x, u)
    k2 = dynamics(x + k1*dt/2, u)
    k3 = dynamics(x + k2*dt/2, u)
    k4 = dynamics(x + k3*dt, u)
    return x + ((dt/6) * (k1+2*k2+2*k3+k4))

    
def simulate(dynamics, params, integrator):
    x_0 = np.vstack([params['q0'], params['qd0']])
    dt = params["dt"]
    N = int((params["T"])/dt)
    T = np.arange(0, params["T"], dt)
    X = np.zeros((N, 4))
    X[0] = x_0.T[0]
    for k in range(N-1):
        u = -params["K"] @ X[k]
        X[k+1] = integrator(X[k], u, dynamics, dt)

    return T, X




# plant =  MultibodyPlant(0.0)
# parser = Parser(plant)
# parser.AddModels("inverse_pendulum.sdf")
# plant.Finalize()

def drake_inv_pendulum_plant(params):
    plant = MultibodyPlant(0.0)

    cart = plant.AddRigidBody("cart", SpatialInertia(params["m_c"], np.zeros(3), UnitInertia(0,0,0)))
    slide = plant.AddJoint(PrismaticJoint("slide", plant.world_frame(), plant.GetFrameByName("cart"), [1,0,0], damping=params["damping_c"]))

    plant.AddJointActuator("slide_ac", slide)
    plant.AddRigidBody("pendulum", SpatialInertia(params["m_p"], np.zeros(3), UnitInertia(0,0,0)))
    f = plant.AddFrame(FixedOffsetFrame(f"f", plant.GetFrameByName("pendulum"), RigidTransform([0,0,-params["l"]])))
    plant.AddJoint(RevoluteJoint(f"pivot", plant.GetFrameByName(f"cart"), f, [0,-1,0], params["damping_p"]))

    plant.Finalize()
    return plant

def linearize_dynamics(dynamics, x_0, u_0):
    dyn = lambda x, u: dynamics(x, u, params)
    A = jacobian(lambda dx: rk4(dx, u_0, dyn, params["dt"]), x_0)
    B = jacobian(lambda du: rk4(x_0, du, dyn, params["dt"]), u_0)
    return A, B



if __name__ == "__main__":
    server = viser.ViserServer(port=8081)

    params = {
        "T": 4,
        "dt": 0.01,
        "q0": np.array([[0, 0.4]]).T,
        "qd0": np.array([[0, 0]]).T,
        "K": np.zeros((1, 4)),
        "Q": np.diag([10,0.1,0.1,0]),
        "R": np.diag([0.001]),
        "damping_c": 0,
        "damping_p": 0,
        "m_c": 1,
        "m_p": 50,
        "l": 1
    }

    start = time.perf_counter()
    plant = drake_inv_pendulum_plant(params)
    params["plant"] = plant
    params["plant_context"] = plant.CreateDefaultContext()
    params["plant_ad"] = plant.ToAutoDiffXd()
    params["plant_ad_context"] = params["plant_ad"].CreateDefaultContext()
    A, B = linearize_dynamics(dynamics, np.zeros(4), np.zeros(1))
    params["K"] = dare(A, B, params["Q"], params["R"])[1]
    T, X = simulate(lambda x,u : dynamics(x, u, params), params, rk4)
    pendulum_trajectory : PiecewisePolynomial = PiecewisePolynomial.FirstOrderHold(T, X.T)
    pendulum_handles = add_pendulum_visual(server.scene, params)

    visualization_fps = 60

    with server.gui.add_folder("Parameters"):
        gui_enable_cb = server.gui.add_checkbox("Enable Controller", True)
        server.gui.add_button("Run Experiment")

    with server.gui.add_folder("Playback"):
        gui_frame_slider = server.gui.add_slider(
            "Time", 0, params["T"], 1 / visualization_fps, 0
        )
        gui_frame_step_buttons = server.gui.add_button_group(
            "Step", ["<<", "<", ">", ">>"]
        )
        gui_play_button = server.gui.add_button(
            "Play", icon=viser.Icon.PLAYER_PLAY_FILLED
        )
        gui_pause_button = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)
        gui_play_speed = server.gui.add_number(
            "Speed", 1.5  # TODO hack to allow floats
        )

    @gui_play_button.on_click
    def _(_) -> None:
        gui_play_button.icon = viser.Icon.PLAYER_PLAY_FILLED
        gui_pause_button.icon = viser.Icon.PLAYER_PAUSE
        if gui_frame_slider.value == gui_frame_slider.max:
            gui_frame_slider.value = 0

    @gui_pause_button.on_click
    def _(_) -> None:
        gui_play_button.icon = viser.Icon.PLAYER_PLAY
        gui_pause_button.icon = viser.Icon.PLAYER_PAUSE_FILLED
    
    
    @gui_frame_step_buttons.on_click
    def _(_) -> None:
        match gui_frame_step_buttons.value:
            case "<<":
                gui_frame_slider.value = max(
                    gui_frame_slider.min,
                    gui_frame_slider.value - 5.0 / visualization_fps,
                )
            case "<":
                gui_frame_slider.value = max(
                    gui_frame_slider.min,
                    gui_frame_slider.value - 1.0 / visualization_fps,
                )
            case ">>":
                gui_frame_slider.value = min(
                    gui_frame_slider.max,
                    gui_frame_slider.value + 5.0 / visualization_fps,
                )
            case ">":
                gui_frame_slider.value = min(
                    gui_frame_slider.max,
                    gui_frame_slider.value + 1.0 / visualization_fps,
                )

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        gui_play_speed.value = 1
        client.scene.world_axes.visible = True

    while True:
        frame_idx = gui_frame_slider.value
        playing = gui_play_button.icon == viser.Icon.PLAYER_PLAY_FILLED
        if not playing and frame_idx == last_frame_idx and not refresh:
            continue
        refresh = False
        last_frame_idx = frame_idx

        if playing:
            gui_frame_slider.value = min(
                gui_frame_slider.max, gui_frame_slider.value + 1.0 / visualization_fps
            )

            if gui_frame_slider.value == gui_frame_slider.max:
                gui_play_button.icon = viser.Icon.PLAYER_PLAY
                gui_pause_button.icon = viser.Icon.PLAYER_PAUSE_FILLED
        
        params["plant"].SetPositionsAndVelocities(params["plant_context"], pendulum_trajectory.value(gui_frame_slider.value))
        set_pendulum_visual(pendulum_handles, params["plant"], params["plant_context"])

        play_speed = 1 if gui_play_speed.value == 0 else gui_play_speed.value
        time.sleep(1 / (play_speed * visualization_fps))