# bnknotes Makefile
.PHONY: all build build-prod install uninstall build-linux deps test test-install test-abi check \
	fix perf perf-save perf-lab tables clean size shaders

# App configuration
APP_NAME := bnknotes

# The triple .cargo/config.toml names, which build-std requires and which puts every
# artifact under target/<triple>/. Building on another architecture means overriding it:
# make build-prod TARGET=aarch64-unknown-linux-gnu
TARGET := x86_64-unknown-linux-gnu
VERSION := $(shell grep -m1 '^version' Cargo.toml | cut -d'"' -f2)
TARBALL := $(APP_NAME)-v$(VERSION)-$(TARGET).tar.gz

# Paths. Everything installs under $HOME: bnknotes is a single-user desktop app, so a
# root-owned copy in /usr/local/bin buys nothing and costs a sudo on every install.
# ~/.local/bin also comes first on PATH, so this is the copy that wins regardless.
BUILD_PATH := target/$(TARGET)/release
BIN_DIR := ~/.local/bin
APPS_DIR := ~/.local/share/applications
ICON_DIR := ~/.local/share/icons/hicolor/scalable/apps

# The oldest glibc a downloaded tarball runs on, which is Debian 12's. The
# release builds in that image and checks the binary against this.
GLIBC_FLOOR := 2.36

# The Unicode Character Database, installed to /usr/share/unicode/. Only `make tables`
# (and refreshing ucd/ on a version bump) needs it; the committed tables build without.
UCD_PKG := unicode-character-database

# Default target
all: test build

# Development build
build:
	cargo build --target $(TARGET)

# Production build
build-prod:
	cargo build --release --target $(TARGET)

# Install the release binary, desktop entry, and icon into the user's session. Never
# needs root (see the note on the paths above). The desktop-database refresh is what
# registers the text/markdown association, so the entry opens .md files after install.
install: build-prod
	install -Dm755 $(BUILD_PATH)/$(APP_NAME) $(BIN_DIR)/$(APP_NAME)
	install -Dm644 assets/$(APP_NAME).desktop $(APPS_DIR)/$(APP_NAME).desktop
	install -Dm644 assets/$(APP_NAME).svg $(ICON_DIR)/$(APP_NAME).svg
	-update-desktop-database $(APPS_DIR)
	@echo "Installed $(APP_NAME) to $(BIN_DIR)/"
	@echo "Installed desktop file to $(APPS_DIR)/"
	@echo "Installed icon to $(ICON_DIR)/"

# Remove everything `install` put in the session.
uninstall:
	rm -f $(BIN_DIR)/$(APP_NAME) $(APPS_DIR)/$(APP_NAME).desktop $(ICON_DIR)/$(APP_NAME).svg
	-update-desktop-database $(APPS_DIR)
	@echo "Removed $(APP_NAME) from $(BIN_DIR)/ and ~/.local/share/"

# Build a release tarball into dist/ for upload to a GitHub Release, with the
# checksum install.sh verifies beside it. The layout is install.sh's contract.
build-linux: build-prod
	sh .github/scripts/check-abi.sh $(GLIBC_FLOOR) $(BUILD_PATH)/$(APP_NAME)
	rm -rf dist/stage
	mkdir -p dist/stage/icons/hicolor/scalable/apps
	cp $(BUILD_PATH)/$(APP_NAME) dist/stage/$(APP_NAME)
	cp assets/$(APP_NAME).desktop dist/stage/$(APP_NAME).desktop
	cp assets/$(APP_NAME).svg dist/stage/icons/hicolor/scalable/apps/$(APP_NAME).svg
	tar czf dist/$(TARBALL) -C dist/stage .
	rm -rf dist/stage
	cd dist && sha256sum $(TARBALL) > $(TARBALL).sha256
	@echo "Built dist/$(TARBALL), with a .sha256"

# Format check, lint, and tests. Clippy runs twice: the test build counts anything a
# test calls as used, so only the shipping build finds code nothing else reaches.
test:
	cargo fmt --check
	cargo clippy --all --benches --tests --examples --all-features -- -D clippy::all -D warnings
	cargo clippy -- -D clippy::all -D warnings
	cargo test

# The installer, over file:// URLs, with no network.
test-install:
	sh .github/scripts/test-install.sh

# The ABI floor check, against binaries this machine already has.
test-abi:
	sh .github/scripts/test-abi.sh

check: test test-install test-abi

# Format and lint
fix:
	cargo fmt
	cargo clippy --all --benches --tests --examples --all-features --fix --allow-dirty

# Recompile the GPU shaders to the committed SPIR-V blobs. Needs glslc, and
# only when a shader source changes; normal builds use the committed blobs.
shaders:
	glslc -O shaders/quad.vert -o shaders/quad.vert.spv
	glslc -O shaders/quad.frag -o shaders/quad.frag.spv

# Measure the Tier 1 gate against perf/baseline.txt (allocations counted too)
perf:
	cargo run --release --features perf-alloc -- --perf

# Accept current perf numbers as the new baseline
perf-save:
	cargo run --release --features perf-alloc -- --perf --save

# Run the Tier 2 lab over the stress corpora, appending to perf/history.jsonl.
# Pass SIZE (MiB per corpus) or CORPUS to narrow it, e.g. `make perf-lab CORPUS=emoji SIZE=8`.
perf-lab:
	cargo run --release --features perf-alloc -- --perf-lab $(if $(SIZE),--size $(SIZE)) $(CORPUS)

# Install the dev-only Unicode Character Database (for `make tables` on a version bump).
deps:
	sudo pacman -S --needed --asexplicit $(UCD_PKG)

# Regenerate the committed Unicode property tables (src/platform/grapheme_tables.rs)
# from the vendored ucd/ data. Deterministic and offline; only needed when ucd/ is
# refreshed for a new Unicode version (`make deps`, then copy /usr/share/unicode/ into
# ucd/). The output is guarded by the conformance test over ucd/GraphemeBreakTest.txt.
tables:
	cargo run --release --features dev -- --gen-tables

# Clean build artifacts
clean:
	cargo clean

# Show binary size
size:
	@echo "Release build:"
	@ls -lh $(BUILD_PATH)/$(APP_NAME) 2>/dev/null || echo "  Not built"
