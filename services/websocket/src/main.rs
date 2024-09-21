use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::accept_async;
use futures_util::{SinkExt, StreamExt};

fn main() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let addr = "0.0.0.0:9001";
        let listener = TcpListener::bind(&addr).await.expect("Failed to bind");
        println!("WebSocket server listening on: {}", addr);

        while let Ok((stream, _)) = listener.accept().await {
            tokio::spawn(handle_connection(stream));
        }
    });
}

async fn handle_connection(stream: TcpStream) {
    let ws_stream = accept_async(stream).await.expect("Error during WebSocket handshake");
    println!("New WebSocket connection: {}", ws_stream.get_ref().peer_addr().unwrap());

    let (mut write, mut read) = ws_stream.split();

    while let Some(message) = read.next().await {
        match message {
            Ok(msg) => {
                println!("Received a message: {}", msg);
                if let Err(e) = write.send(msg).await {
                    eprintln!("Error sending message: {}", e);
                    break;
                }
            }
            Err(e) => {
                eprintln!("Error receiving message: {}", e);
                break;
            }
        }
    }

    println!("WebSocket connection closed");
}
