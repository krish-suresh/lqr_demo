from pendulum import *
from visualization import *



def main():
    clients = {} # by client id
    server = viser.ViserServer(port=8081)
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.scene.world_axes.visible = True
        clients[client.client_id] = client
    
    


if __name__ == "__main__":
    main()