# Rust dev environment for the pure-Rust port (free_form_.../rust).
# No source is COPYed in — the workspace is bind-mounted at runtime; this image
# only provides the toolchain. rust:slim already ships gcc + libc6-dev for
# linking, so only git/zsh and the rustup components are added here.
FROM rust:1-slim-bookworm

# git: the crate fetches geo-booleanop from a pinned git rev, so cargo needs
# git available. curl + ca-certificates back the oh-my-zsh installer below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    zsh \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Editor tooling: rust-analyzer powers the VS Code extension; clippy/rustfmt
# back the usual lint/format commands.
RUN rustup component add rust-analyzer clippy rustfmt

RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
