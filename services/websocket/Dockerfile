# Use the official Rust image
FROM rust:1.72 AS builder

# Set the working directory
WORKDIR /src

# Copy the Cargo.toml and Cargo.lock files
COPY Cargo.toml Cargo.lock ./

# Create an empty project to cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -rf src

# Copy the source code
COPY src ./src

# Build the WebSocket server
RUN cargo build --release

# Use a smaller base image for the final binary
FROM debian:bullseye-slim

# Install necessary runtime dependencies
RUN apt-get update && apt-get install -y libssl1.1 ca-certificates && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the binary from the builder stage
COPY --from=builder /src/target/release/websocket /app/websocket

# Expose the WebSocket port
EXPOSE 9001

# Run the WebSocket server
CMD ["./websocket"]