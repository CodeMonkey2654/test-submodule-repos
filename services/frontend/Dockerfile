# Use the official Bun image
FROM oven/bun:latest AS base
WORKDIR /src/app

# Copy package.json first to leverage Docker cache
COPY package.json .
COPY bun.lockb .

FROM base AS install
RUN bun install
COPY . .
RUN bun run build

# Production stage
FROM base AS release
WORKDIR /src/app

# Copy built assets and package.json from the install stage
COPY --from=install /src/app/dist ./dist
COPY --from=install /src/app/package.json .

# Install only production dependencies
RUN bun install --production
# Install Vite globally
RUN bun install -g vite

# Ensure the start script is available
RUN if ! grep -q '"start"' package.json; then \
    sed -i 's/"scripts": {/"scripts": {\n    "start": "vite preview --host",/' package.json; \
fi


# Expose the port the app runs on
EXPOSE 3000

# Start the application
CMD ["bun", "run", "start"]