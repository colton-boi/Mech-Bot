FROM rust:1-alpine AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --release --locked -p bot && strip /src/target/release/bot

FROM scratch
COPY --from=build /src/target/release/bot /bot
ENTRYPOINT ["/bot"]
