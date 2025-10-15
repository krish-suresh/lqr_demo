import viser
import numpy as np

def main():
    server = viser.ViserServer(port=8081)

    server.scene.world_axes.visible = True
    @server.on_client_connect
    def handle_client(client: viser.ClientHandle):
        client.scene.add_frame("/unique_frame", position=(np.random.randn(), 0, 0))

    server.sleep_forever()

if __name__ == "__main__":
    main()
