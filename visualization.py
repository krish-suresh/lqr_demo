import viser
from pydrake.all import *
import trimesh


def add_pendulum_visual(scene: viser.SceneApi, params) -> tuple[viser.BoxHandle, viser.IcosphereHandle, viser.BoxHandle]:
    cart_size = np.array([0.5, 0.25, 0.25])
    pendulum_size = 0.1
    stick_size = np.array([0.05, 0.05, params["l"]])
    cart = scene.add_box("cart", color=(0, 255, 0), dimensions=cart_size, opacity=0.5)
    pendulum = scene.add_icosphere("pendulum", pendulum_size, color=(0, 255, 0))
    stick = scene.add_box("stick", color=(0, 255, 0), dimensions=stick_size)
    return cart, pendulum, stick

def set_pendulum_visual(pendulum_handles: tuple[viser.BoxHandle, viser.IcosphereHandle, viser.BoxHandle], plant: MultibodyPlant, plant_context: Context):
    X_WC = plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetFrameByName("cart"))
    X_WP = plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetFrameByName("pendulum"))
    pendulum_handles[0].position = X_WC.translation()
    pendulum_handles[0].wxyz = X_WC.rotation().ToQuaternion().wxyz()

    pendulum_handles[1].position = X_WP.translation()
    
    pendulum_handles[2].position = (X_WP.translation() + X_WC.translation())/2.0
    pendulum_handles[2].wxyz = X_WP.rotation().ToQuaternion().wxyz()