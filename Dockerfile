# The submitted bot. Build from this folder: docker build -t mech-rs .
# A match runs it as uid 65534 with no network, a read-only root, 1 CPU, 512 MB,
# and the map at /arena. The binary is static, so the runtime image is empty.
FROM rust:1.100.0-alpine3.22 AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --release --locked -p bot

FROM scratch
COPY --from=build /src/target/release/bot /bot
ENTRYPOINT ["/bot"]
