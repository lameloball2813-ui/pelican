# Pelican

Pelican is a fork of [VIctory](https://github.com/Marvisak/VIctory). This file
lists where the two differ. Everything not listed here is VIctory's, unchanged.

## Differences from VIctory

| Area | Pelican | VIctory |
|------|---------|---------|
| Executable | `pelican.exe` | `victory.exe` |
| Default data and cache folder | `%LOCALAPPDATA%\pelican` | `%LOCALAPPDATA%\victory` |
| Input | Keyboard reports the character each held key types (layout, Shift, Caps Lock, AltGr, dead keys) | Character fields left empty |

Kept on purpose, so upstream changes merge without conflicts:

- package and crate names (`victory`, `victory-*`), so build with
  `cargo build --release -p victory`;
- the `VICTORY_*` environment variables and `victory.toml`;
- the asset policy: no game files, dumps or anything derived from the title
  in this repository, checked on every commit (`cargo xtask install-hooks`).

## Taking VIctory's changes

```powershell
git remote add upstream https://github.com/Marvisak/VIctory.git   # once
git fetch upstream
git merge upstream/main
cargo test --workspace
```

## Licence

GPL-3.0-or-later, as VIctory. See [LICENSE](LICENSE). Copyright in VIctory's
code stays with its authors.
