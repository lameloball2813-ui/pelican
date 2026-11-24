# Pelican

Pelican is a fork of [VIctory](https://github.com/Marvisak/VIctory). This file
lists where the two differ. Everything not listed here is VIctory's, unchanged.

## Differences from VIctory

| Area | Pelican | VIctory |
|------|---------|---------|
| Executable | `pelican.exe` | `victory.exe` |
| Default data and cache folder | `%LOCALAPPDATA%\pelican` | `%LOCALAPPDATA%\victory` |
| Input | Keyboard reports the character each held key types (layout, Shift, Caps Lock, AltGr, dead keys) | Character fields left empty |

## Fixes waiting on VIctory, merged here

VIctory accepts pull requests from collaborators only, so fixes offered on
its issues wait on contributors' forks. Pelican takes them in as their
authors' own commits (each says `cherry picked from commit ...`), after
reading them and running the tests on an RTX 5060 Ti and WARP.

| VIctory issue | Fix | Author |
|---|---|---|
| #761 | architecture.md lists BCPack as implemented | geiyiliyabumo |
| #792 | `VICTORY_CAPTURE` makes its folder if missing | JFIREonFIRE |
| #727 | display deferral works where Windows refuses the power request | geiyiliyabumo |
| #831 | `pelican config` fails when a path breaks the asset policy | kyle7908 |
| #718 | format support and multisample queries answered by the host GPU | fuzeofficlal |
| #832 | float `v_cmp_f`/`v_cmp_tru` no longer emit `fcmp false/true` (removed WARP) | Peppacodes |
| #830 | `v_cmp_class_f32` modifiers keep denormals and signalling NaNs | AndyLatimer |
| #823 | 16-bit SDWA `WORD_0` with `UNUSED_SEXT` sign-extends | AndyLatimer |
| #775 | `v_swaprel_b32`, `v_movrelsd_2_b32` lowered | talakasse |
| #773 | automatically sized textures report their mip count | Xvania7 |
| #782 | `GetResourceAllocationInfo` alignments and overflow | AndyLatimer |
| #779 | `GetDesc` methods return their output pointer | BlueCatsz |
| #65 | sampler heaps over 2,048 keep the host heap | Xvania7 |
| #777 | DXBC parser refuses parts inside its header | Xvania7 |
| #805 | data-file loads reach the host loader (tightened in Pelican: code loads stay refused) | davutsimsekk |
| #793 | asset policy folds trailing `..` and `\\?\` paths | geiyiliyabumo |
| #790 | the space below 4 GiB is kept for the title's 32-bit memory | NuraadGit |

Pelican's own fixes for VIctory issues: #811 (OLEAUT32 BSTR functions) and
hardware tests for #800, #802, #803.

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
