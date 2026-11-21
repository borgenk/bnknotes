# BNK Notes

[![CI](https://github.com/borgenk/bnknotes/actions/workflows/ci.yml/badge.svg)](https://github.com/borgenk/bnknotes/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/borgenk/bnknotes)](https://github.com/borgenk/bnknotes/releases/latest)

A native Wayland + Vulkan markdown notes editor.

_Disclaimer: personal project, built mainly for my own use, AI-assisted._

## Requirements

- A Linux desktop running a **Wayland** session
- A working **Vulkan** device
- The freetype, harfbuzz, fontconfig and libxkbcommon libraries

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/borgenk/bnknotes/main/install.sh | sh
```

This drops the binary in `~/.local/bin` and installs the desktop entry and icon, so
markdown files open in it from the file manager. From a checkout, `make install` does
the same.

## Configuration

The config file and the session state live under the XDG config dir
(`$XDG_CONFIG_HOME/bnknotes`, falling back to `~/.config/bnknotes`):

- `config` - the vaults, one `vault <path>` line each
- `state.bin` - open tabs, active vault, and window layout
