//! Rule #1, enforced at runtime: game-derived data never lands in the repo.
//!
//! Everything VIctory produces from the user's files (converted assets,
//! prepared images, shader caches, crash dumps) is as much "from the game"
//! as the files themselves. Every tool that writes such output must call
//! [`ensure_outside_repo`] on the destination first. The static half of the
//! rule is `cargo xtask policy`, which checks what git tracks.

use std::path::{Component, Path, PathBuf};

/// Marker file at the root of the source tree.
pub const ROOT_MARKER: &str = ".victory-root";

/// The source tree this binary was compiled from.
fn compiled_workspace_root() -> PathBuf {
    // crates/victory-core -> workspace root
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

#[derive(Debug, thiserror::Error)]
pub enum PolicyError {
    #[error(
        "refusing to use {path} for game-derived data: it is inside the VIctory source tree \
         ({root}). Keep game files, caches and outputs outside the repository \
         (see docs/asset-policy.md)"
    )]
    InsideRepo { path: PathBuf, root: PathBuf },
    #[error("cannot resolve {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
}

/// Returns `Ok` only if `path` is outside every VIctory source tree.
///
/// `path` does not have to exist yet. Two checks are made:
/// 1. it is not below the source tree this binary was built from;
/// 2. none of its ancestors contains the [`ROOT_MARKER`] file (catches
///    other clones/copies of the repository).
pub fn ensure_outside_repo(path: &Path) -> Result<(), PolicyError> {
    let resolved = resolve(path)?;

    if let Ok(root) = std::fs::canonicalize(compiled_workspace_root())
        && root.join(ROOT_MARKER).is_file()
        && resolved.starts_with(&root)
    {
        return Err(PolicyError::InsideRepo {
            path: path.to_path_buf(),
            root,
        });
    }

    for ancestor in resolved.ancestors() {
        if ancestor.join(ROOT_MARKER).is_file() {
            return Err(PolicyError::InsideRepo {
                path: path.to_path_buf(),
                root: ancestor.to_path_buf(),
            });
        }
    }
    Ok(())
}

/// Canonicalizes the longest existing prefix of `path` and folds the
/// non-existing tail onto it, so paths that do not exist yet can still be
/// compared reliably.
///
/// The prefix is peeled one component at a time: `canonicalize` needs every
/// directory on the way to exist, while [`Path::file_name`] names neither `..`
/// nor a trailing `.`, so either one used to end the walk early. Folding the
/// peeled tail directly onto the canonical prefix (instead of pushing `".."`
/// and re-parsing) also handles Windows verbatim prefixes (`\\?\`), where
/// `Path::components` treats `".."` as a normal segment rather than
/// [`Component::ParentDir`].
fn resolve(path: &Path) -> Result<PathBuf, PolicyError> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|source| PolicyError::Io {
                path: path.to_path_buf(),
                source,
            })?
            .join(path)
    };

    let mut existing = absolute.as_path();
    let mut rest: Vec<Component<'_>> = Vec::new();
    loop {
        match std::fs::canonicalize(existing) {
            Ok(canon) => return Ok(append_normalized(canon, &rest)),
            Err(source) => {
                // `components` yields every trailing segment, including `..`
                // and `.` (for which `file_name` returns `None`). A root or
                // prefix is not peelable, which ends the walk: an absent drive
                // never canonicalizes, and this error is its answer.
                let mut components = existing.components();
                match components.next_back() {
                    Some(Component::CurDir) => existing = components.as_path(),
                    Some(component @ (Component::Normal(_) | Component::ParentDir)) => {
                        rest.push(component);
                        existing = components.as_path();
                    }
                    _ => {
                        return Err(PolicyError::Io {
                            path: path.to_path_buf(),
                            source,
                        });
                    }
                }
            }
        }
    }
}

/// Applies the peeled non-existing tail (stored in reverse order) onto a
/// canonicalized prefix, resolving `.` and `..` lexically even when `out` has
/// a Windows `\\?\` verbatim prefix.
fn append_normalized(mut out: PathBuf, reversed_tail: &[Component<'_>]) -> PathBuf {
    for component in reversed_tail.iter().rev() {
        match component {
            Component::CurDir => {}
            Component::Normal(s) if *s == "." => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(s) if *s == ".." => {
                out.pop();
            }
            c => out.push(c.as_os_str()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_paths_inside_this_repo() {
        let inside = compiled_workspace_root().join("cache").join("textures");
        assert!(matches!(
            ensure_outside_repo(&inside),
            Err(PolicyError::InsideRepo { .. })
        ));
        let sneaky = std::env::temp_dir().join("..").join("x"); // sanity: no panic
        let _ = ensure_outside_repo(&sneaky);
    }

    /// `base` with `parts` appended as text, separators and all: `Path::join`
    /// folds a trailing `..` into `base` itself, while these shapes have to
    /// reach [`resolve`] as written, the way they arrive from a config file or
    /// a command line.
    ///
    /// The callers canonicalize `base` first, so on Windows it carries the
    /// `\\?\` prefix: Win32 does not fold a verbatim path either, which is
    /// what keeps `<missing>/..` a path that does not exist on disk.
    fn below(base: &Path, parts: &[&str]) -> PathBuf {
        let mut text = base.display().to_string();
        for part in parts {
            text.push(std::path::MAIN_SEPARATOR);
            text.push_str(part);
        }
        PathBuf::from(text)
    }

    #[test]
    fn rejects_dotdot_escapes_back_into_repo() {
        let root = std::fs::canonicalize(compiled_workspace_root()).unwrap();
        let path = below(&root, &["crates", "..", "not-yet-created", "out"]);
        assert!(matches!(
            ensure_outside_repo(&path),
            Err(PolicyError::InsideRepo { .. })
        ));
    }

    /// `..` (or `.`) after a directory that does not exist yet: `Path` has no
    /// `file_name` for either, so `resolve` used to give up with
    /// `PolicyError::Io` instead of naming the policy the path broke.
    #[test]
    fn trailing_dotdot_inside_this_repo_is_refused_as_such() {
        let root = std::fs::canonicalize(compiled_workspace_root()).unwrap();
        let inside = [
            below(&root, &["not-yet-created", ".."]),
            below(&root, &["crates", "not-yet-created", "..", ".."]),
            below(&root, &["not-yet-created", "."]),
        ];
        for path in inside {
            let result = ensure_outside_repo(&path);
            assert!(
                matches!(result, Err(PolicyError::InsideRepo { .. })),
                "{path:?} -> {result:?}"
            );
        }
    }

    /// The same shapes outside every source tree have to resolve, not fail.
    #[test]
    fn trailing_dotdot_outside_every_repo_resolves() {
        let dir = tempfile::tempdir().unwrap();
        let tmp = std::fs::canonicalize(dir.path()).unwrap();
        let outside = [
            below(&tmp, &["not-yet-created", ".."]),
            below(&tmp, &["not-yet-created", "sub", "..", ".."]),
            below(&tmp, &["not-yet-created", "."]),
        ];
        for path in outside {
            let result = ensure_outside_repo(&path);
            assert!(result.is_ok(), "{path:?} -> {result:?}");
        }
    }

    /// Nothing below a root can be peeled and an absent drive never
    /// canonicalizes, so the walk has to stop and report it.
    #[cfg(windows)]
    #[test]
    fn an_absent_drive_is_an_io_error() {
        if Path::new("Z:\\").exists() {
            return; // this machine has a Z: drive, so it cannot be absent
        }
        assert!(matches!(
            ensure_outside_repo(Path::new("Z:\\not-yet-created\\..")),
            Err(PolicyError::Io { .. })
        ));
    }

    #[test]
    fn rejects_other_clones_via_marker() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(ROOT_MARKER), "").unwrap();
        assert!(ensure_outside_repo(&tmp.path().join("cache")).is_err());
    }

    /// Regression test for Windows verbatim (`\\?\`) paths returned by
    /// `canonicalize`: `..` after a non-existing sibling directory must fold
    /// back into the real target so an inside-repo destination cannot slip
    /// past `starts_with` / `ancestors`, and an escape back outside resolves.
    #[test]
    fn dotdot_after_missing_segment_normalizes_through_verbatim_prefixes() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("clone");
        std::fs::create_dir(&repo).unwrap();
        std::fs::write(repo.join(ROOT_MARKER), "").unwrap();

        // Non-verbatim path entering the marked repo via `<missing>/../clone/cache`.
        let sneaking_in = below(tmp.path(), &["not-yet-created", "..", "clone", "cache"]);
        let result = ensure_outside_repo(&sneaking_in);
        assert!(
            matches!(result, Err(PolicyError::InsideRepo { .. })),
            "{sneaking_in:?} -> {result:?}"
        );

        // Escaping out of the marked repo via `<missing>/../../outside`.
        let escaping_out = below(&repo, &["not-yet-created", "..", "..", "outside"]);
        let result = ensure_outside_repo(&escaping_out);
        assert!(result.is_ok(), "{escaping_out:?} -> {result:?}");
    }

    #[test]
    fn accepts_unrelated_directories() {
        let tmp = tempfile::tempdir().unwrap();
        ensure_outside_repo(&tmp.path().join("does").join("not").join("exist")).unwrap();
    }
}
