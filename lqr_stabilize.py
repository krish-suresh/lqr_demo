from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import viser
from pydrake.all import PiecewisePolynomial, jacobian

from pendulum import dare, dynamics, drake_inv_pendulum_plant, rk4, simulate
from visualization import add_pendulum_visual, set_pendulum_visual

VISUALIZATION_FPS = 60.0
MIN_DURATION = 1e-3
MIN_DT = 1e-4
MIN_MASS = 1e-4
MIN_LENGTH = 1e-4
MIN_R = 1e-6


@dataclass
class ClientState:
    client: viser.ClientHandle
    params: Dict[str, Any]
    pendulum_handles: tuple
    controls: Dict[str, Any] = field(default_factory=dict)
    playback: Dict[str, Any] = field(default_factory=dict)
    trajectory: Optional[PiecewisePolynomial] = None
    sim_times: Optional[np.ndarray] = None
    sim_states: Optional[np.ndarray] = None
    playing: bool = False
    last_frame_value: float = -1.0
    refresh: bool = True


def default_params() -> Dict[str, Any]:
    return {
        "T": 4.0,
        "dt": 0.01,
        "q0": np.array([[0.0], [0.4]]),
        "qd0": np.array([[0.0], [0.0]]),
        "K": np.zeros((1, 4)),
        "Q": np.diag([10.0, 0.1, 0.1, 0.0]),
        "R": np.diag([0.001]),
        "damping_c": 0.0,
        "damping_p": 0.0,
        "m_c": 1.0,
        "m_p": 50.0,
        "l": 1.0,
    }


def rebuild_plants(params: Dict[str, Any]) -> None:
    plant = drake_inv_pendulum_plant(params)
    params["plant"] = plant
    params["plant_context"] = plant.CreateDefaultContext()
    plant_ad = plant.ToAutoDiffXd()
    params["plant_ad"] = plant_ad
    params["plant_ad_context"] = plant_ad.CreateDefaultContext()


def compute_lqr_gain_for_params(params: Dict[str, Any]) -> np.ndarray:
    dyn = lambda x, u: dynamics(x, u, params)
    x0 = np.zeros(4)
    u0 = np.zeros(1)
    A = jacobian(lambda x: rk4(x, u0, dyn, params["dt"]), x0)
    B = jacobian(lambda u: rk4(x0, u, dyn, params["dt"]), u0)
    _, K = dare(A, B, params["Q"], params["R"])
    return K


def update_params_from_controls(state: ClientState) -> None:
    controls = state.controls
    params = state.params

    def _clamp(handle, minimum: float) -> float:
        value = float(handle.value)
        if value < minimum:
            value = minimum
            handle.value = value
        return value

    params["T"] = _clamp(controls["T"], MIN_DURATION)
    params["dt"] = _clamp(controls["dt"], MIN_DT)

    params["q0"] = np.array(
        [[float(controls["q0_cart"].value)], [float(controls["q0_theta"].value)]]
    )
    params["qd0"] = np.array(
        [[float(controls["qd0_cart"].value)], [float(controls["qd0_theta"].value)]]
    )

    params["Q"] = np.diag(
        [
            float(controls["Q_0"].value),
            float(controls["Q_1"].value),
            float(controls["Q_2"].value),
            float(controls["Q_3"].value),
        ]
    )
    params["R"] = np.array([[ _clamp(controls["R_0"], MIN_R) ]])

    params["damping_c"] = float(controls["damping_c"].value)
    params["damping_p"] = float(controls["damping_p"].value)
    params["m_c"] = _clamp(controls["m_c"], MIN_MASS)
    params["m_p"] = _clamp(controls["m_p"], MIN_MASS)
    params["l"] = _clamp(controls["l"], MIN_LENGTH)


def set_status_message(state: ClientState, message: str) -> None:
    status = state.controls.get("status_markdown")
    if status is not None:
        status.content = message


def create_param_controls(state: ClientState) -> None:
    client = state.client
    params = state.params
    controls = state.controls

    controls["run_button"] = client.gui.add_button("Run Simulation")
    controls["status_markdown"] = client.gui.add_markdown(
        "Status: Press **Run Simulation** to generate a trajectory."
    )

    with client.gui.add_folder("Simulation") as f:
        controls["T"] = client.gui.add_number("Duration (T)", float(params["T"]))
        controls["dt"] = client.gui.add_number("Time Step (dt)", float(params["dt"]))
        f.expand_by_default = False

    with client.gui.add_folder("Initial State") as f:
        controls["q0_cart"] = client.gui.add_number("Cart Position", float(params["q0"][0, 0]))
        controls["q0_theta"] = client.gui.add_number("Pendulum Angle", float(params["q0"][1, 0]))
        controls["qd0_cart"] = client.gui.add_number("Cart Velocity", float(params["qd0"][0, 0]))
        controls["qd0_theta"] = client.gui.add_number("Pendulum Velocity", float(params["qd0"][1, 0]))
        f.expand_by_default = False

    with client.gui.add_folder("LQR") as f:
        controls["enable_controller"] = client.gui.add_checkbox("Enable Controller", True)
        controls["Q_0"] = client.gui.add_slider(
            "Q[0,0]", 0.0, 200.0, 0.1, float(params["Q"][0, 0])
        )
        controls["Q_1"] = client.gui.add_slider(
            "Q[1,1]", 0.0, 200.0, 0.1, float(params["Q"][1, 1])
        )
        controls["Q_2"] = client.gui.add_slider(
            "Q[2,2]", 0.0, 200.0, 0.1, float(params["Q"][2, 2])
        )
        controls["Q_3"] = client.gui.add_slider(
            "Q[3,3]", 0.0, 200.0, 0.1, float(params["Q"][3, 3])
        )
        controls["R_0"] = client.gui.add_slider(
            "R[0,0]", MIN_R, 10.0, 0.001, float(params["R"][0, 0])
        )
        f.expand_by_default = False

    with client.gui.add_folder("Physical") as f:
        f.expand_by_default = False
        controls["damping_c"] = client.gui.add_slider(
            "Cart Damping", 0.0, 5.0, 0.01, float(params["damping_c"])
        )
        controls["damping_p"] = client.gui.add_slider(
            "Pivot Damping", 0.0, 5.0, 0.01, float(params["damping_p"])
        )
        controls["m_c"] = client.gui.add_slider(
            "Cart Mass", MIN_MASS, 100.0, 0.01, float(params["m_c"])
        )
        controls["m_p"] = client.gui.add_slider(
            "Pendulum Mass", MIN_MASS, 100.0, 0.01, float(params["m_p"])
        )
        controls["l"] = client.gui.add_slider(
            "Pendulum Length", MIN_LENGTH, 10.0, 0.01, float(params["l"])
        )


def create_playback_controls(state: ClientState) -> None:
    client = state.client
    playback = state.playback
    params = state.params

    with client.gui.add_folder("Playback"):
        playback["slider"] = client.gui.add_slider(
            "Time", 0.0, float(params["T"]), 1.0 / VISUALIZATION_FPS, 0.0
        )
        playback["step_buttons"] = client.gui.add_button_group("Step", ["<<", "<", ">", ">>"])
        playback["play_button"] = client.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
        playback["pause_button"] = client.gui.add_button(
            "Pause", icon=viser.Icon.PLAYER_PAUSE_FILLED
        )
        playback["play_speed"] = client.gui.add_number("Speed", 1.0)


def update_play_pause_icons(state: ClientState, playing: bool) -> None:
    play_button = state.playback["play_button"]
    pause_button = state.playback["pause_button"]
    if playing:
        play_button.icon = viser.Icon.PLAYER_PLAY_FILLED
        pause_button.icon = viser.Icon.PLAYER_PAUSE
    else:
        play_button.icon = viser.Icon.PLAYER_PLAY
        pause_button.icon = viser.Icon.PLAYER_PAUSE_FILLED
    state.playing = playing


def run_simulation(state: ClientState) -> None:
    set_status_message(state, "Running simulation")
    update_params_from_controls(state)
    rebuild_plants(state.params)
    K = compute_lqr_gain_for_params(state.params)

    enable_controller = state.controls["enable_controller"].value
    local_K = K if enable_controller else np.zeros_like(K)
    previous_K = state.params.get("K")

    try:
        state.params["K"] = local_K
        T, X = simulate(lambda x, u: dynamics(x, u, state.params), state.params, rk4)

        state.sim_times = T
        state.sim_states = X
        state.trajectory = PiecewisePolynomial.FirstOrderHold(T, X.T)

        slider = state.playback["slider"]
        slider.max = float(state.params["T"])
        slider.min = 0.0
        slider.step = 1.0 / VISUALIZATION_FPS
        slider.value = 0.0

        state.last_frame_value = -1.0
        update_play_pause_icons(state, True)
        state.refresh = True

        state.params["plant"].SetPositionsAndVelocities(
            state.params["plant_context"], state.trajectory.value(0.0)
        )
        set_pendulum_visual(state.pendulum_handles, state.params["plant"], state.params["plant_context"])
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            state.params["K"] = previous_K if previous_K is not None else K
            raise
        state.params["K"] = previous_K if previous_K is not None else K
        slider = state.playback.get("slider")
        if slider is not None:
            slider.value = slider.min
        update_play_pause_icons(state, False)
        state.playing = False
        state.refresh = False
        state.trajectory = None
        state.sim_times = None
        state.sim_states = None
        set_status_message(
            state,"Simulation failed. Please refresh the page before trying again."
        )
        print(f"[error] Simulation failed for client {state.client.client_id}: {exc}")
        return
    else:
        state.params["K"] = K
        set_status_message(state, "Simulation running.")


def bind_callbacks(state: ClientState) -> None:
    @state.controls["run_button"].on_click
    def _(_event) -> None:
        run_simulation(state)

    @state.playback["play_button"].on_click
    def _(_event) -> None:
        slider = state.playback["slider"]
        if slider.value >= slider.max:
            slider.value = slider.min
        update_play_pause_icons(state, True)
        state.refresh = True

    @state.playback["pause_button"].on_click
    def _(_event) -> None:
        update_play_pause_icons(state, False)

    @state.playback["step_buttons"].on_click
    def _(_event) -> None:
        slider = state.playback["slider"]
        step = getattr(slider, "step", 1.0 / VISUALIZATION_FPS)
        selection = state.playback["step_buttons"].value
        if selection == "<<":
            slider.value = max(slider.min, slider.value - 5.0 * step)
        elif selection == "<":
            slider.value = max(slider.min, slider.value - step)
        elif selection == ">>":
            slider.value = min(slider.max, slider.value + 5.0 * step)
        elif selection == ">":
            slider.value = min(slider.max, slider.value + step)
        update_play_pause_icons(state, False)
        state.refresh = True


def update_client_simulation(state: ClientState) -> None:
    if state.trajectory is None:
        return

    slider = state.playback["slider"]
    frame_value = slider.value

    if state.playing:
        play_speed = state.playback["play_speed"].value
        if play_speed <= 0:
            play_speed = 1.0
        increment = play_speed / VISUALIZATION_FPS
        new_value = min(slider.max, frame_value + increment)
        if not np.isclose(new_value, frame_value):
            slider.value = new_value
            frame_value = new_value
            state.refresh = True
        if np.isclose(frame_value, slider.max):
            slider.value = slider.max
            state.refresh = True
            update_play_pause_icons(state, False)

    if not state.refresh and np.isclose(frame_value, state.last_frame_value):
        return

    state.refresh = False
    state.last_frame_value = frame_value
    state.params["plant"].SetPositionsAndVelocities(
        state.params["plant_context"], state.trajectory.value(frame_value)
    )
    set_pendulum_visual(state.pendulum_handles, state.params["plant"], state.params["plant_context"])


def main() -> None:
    server = viser.ViserServer(port=8081)
    clients: Dict[int, ClientState] = {}

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = np.array([1.41770366, 1.87300509, 0.33091806])
        client.camera.wxyz = np.array([-0.17767678,  0.24119711,  0.76815397, -0.56585722])
        client.camera.look_at = np.zeros(3)
        client.scene.world_axes.visible = True
        params = default_params()
        rebuild_plants(params)
        pendulum_handles = add_pendulum_visual(client.scene, params)
        state = ClientState(client=client, params=params, pendulum_handles=pendulum_handles)
        create_param_controls(state)
        create_playback_controls(state)
        bind_callbacks(state)
        clients[client.client_id] = state
        run_simulation(state)

    @server.on_client_disconnect
    def _(client: viser.ClientHandle) -> None:
        clients.pop(client.client_id, None)

    try:
        while True:
            for state in list(clients.values()):
                update_client_simulation(state)
            time.sleep(1.0 / VISUALIZATION_FPS)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
