import numpy as np

from pydrake.all import *
from pendulum import *



def generate_swingup(params):
    plant = params["plant"]

    port = plant.get_actuation_input_port().get_index()

    dt_min = params.get("dircol_dt_min", 0.005)
    dt_max = params.get("dircol_dt_max", 0.05)
    N = params.get("dircol_num_samples", 201)

    dircol = DirectCollocation(
        plant,
        plant.CreateDefaultContext(),
        N,
        dt_min,    
        dt_max,
        port
    )
    prog = dircol.prog()
    dircol.AddEqualTimeIntervalsConstraints()

    x0 = np.vstack((params["q0"], params["qd0"])).flatten()
    xg = params.get("x_goal", np.zeros(4))
    prog.AddBoundingBoxConstraint(x0, x0, dircol.initial_state())
    prog.AddBoundingBoxConstraint(xg, xg, dircol.final_state())

    Q = params.get("Q_dircol", params["Q"])
    R = params.get("R_dircol", params["R"])
    for k in range(N):
        prog.AddQuadraticErrorCost(Q, xg, dircol.state(k))
    for k in range(N - 1):
        prog.AddQuadraticCost(R, np.zeros(R.shape[0]), dircol.input(k))
    dircol.AddFinalCost(0.01 * dircol.time())

    T_guess = float(params.get("T", 4.0))
    initial_x_traj = PiecewisePolynomial.FirstOrderHold(
        [0.0, T_guess], np.column_stack((x0, xg))
    )
    initial_u_traj = PiecewisePolynomial.ZeroOrderHold(
        [0.0, T_guess], np.zeros((1, 2))
    )
    dircol.SetInitialTrajectory(initial_u_traj, initial_x_traj)

    solver = SnoptSolver()

    result = solver.Solve(prog)
    if not result.is_success():
        print("Direct collocation failed with", result.get_solution_result())
        raise RuntimeError("Swingup direct collocation was infeasible.")

    X_soln = dircol.ReconstructStateTrajectory(result)
    U_soln = dircol.ReconstructInputTrajectory(result)
    print(result.get_solution_result())
    print("Solver is ", result.get_solver_id().name())
    return U_soln

def simulate_control(dynamics, params, integrator, U: Trajectory):
    x_0 = np.vstack([params['q0'], params['qd0']])
    dt = params["dt"]
    N = int((params["T"])/dt)
    T = np.arange(0, params["T"], dt)
    X = np.zeros((N, 4))
    X[0] = x_0.T[0]
    for k in range(N-1):
        u = U.value(k*dt)
        X[k+1] = integrator(X[k], u, dynamics, dt)

    return T, X


if __name__ == "__main__":

    params = {
        "T": 4,
        "dt": 0.01,
        "q0": np.array([[0, np.pi]]).T,
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


    U: Trajectory = generate_swingup(params)
    params["T"] = U.end_time()

    T, X = simulate_control(lambda x,u : dynamics(x, u, params), params, rk4, U)

    pendulum_trajectory : PiecewisePolynomial = PiecewisePolynomial.FirstOrderHold(T[:-1], X.T)

    # Visualize
    server = viser.ViserServer(port=8081)
    pendulum_handles = add_pendulum_visual(server.scene, params)

    visualization_fps = 60

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
