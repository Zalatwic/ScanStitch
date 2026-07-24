//! Same-directory staged file replacement.
//!
//! Writers encode into a temporary file beside the destination and expose the
//! destination only after the writer succeeds. This gives each individual file
//! an atomic replacement boundary on the platforms supported by `tempfile`.
//! It is deliberately not a cross-artifact transaction or a power-loss
//! durability guarantee: file and directory metadata are not explicitly
//! synchronized before/after the rename.

use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use tempfile::{Builder, NamedTempFile};

pub const ATOMIC_COMMIT_STRATEGY: &str = "same_directory_temporary_file_atomic_replace";

/// A destination-bound temporary file that is removed if it is dropped before commit.
pub struct AtomicFile {
    destination: PathBuf,
    temporary: NamedTempFile,
}

impl AtomicFile {
    pub fn new(destination: &Path) -> io::Result<Self> {
        let parent = destination
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let temporary = Builder::new()
            .prefix(".scanstitch-atomic-")
            .suffix(".tmp")
            .tempfile_in(parent)?;
        Ok(Self {
            destination: destination.to_path_buf(),
            temporary,
        })
    }

    pub fn file_mut(&mut self) -> &mut File {
        self.temporary.as_file_mut()
    }

    /// Atomically replace the destination after flushing userspace buffers.
    pub fn commit(mut self) -> io::Result<()> {
        self.temporary.as_file_mut().flush()?;
        self.temporary
            .persist(&self.destination)
            .map(|_| ())
            .map_err(|error| error.error)
    }

    /// Atomically create the destination, failing if it already exists.
    pub fn commit_noclobber(mut self) -> io::Result<()> {
        self.temporary.as_file_mut().flush()?;
        self.temporary
            .persist_noclobber(&self.destination)
            .map(|_| ())
            .map_err(|error| error.error)
    }
}

pub fn write_bytes(path: &Path, contents: &[u8]) -> io::Result<()> {
    let mut staged = AtomicFile::new(path)?;
    staged.file_mut().write_all(contents)?;
    staged.commit()
}

pub fn write_bytes_noclobber(path: &Path, contents: &[u8]) -> io::Result<()> {
    let mut staged = AtomicFile::new(path)?;
    staged.file_mut().write_all(contents)?;
    staged.commit_noclobber()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_staged_file_atomically_replaces_existing_destination() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("artifact.bin");
        std::fs::write(&destination, b"old complete bytes").unwrap();

        write_bytes(&destination, b"new complete bytes").unwrap();

        assert_eq!(std::fs::read(&destination).unwrap(), b"new complete bytes");
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    }

    #[test]
    fn dropped_staged_file_preserves_existing_destination_and_cleans_temporary_file() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("artifact.bin");
        std::fs::write(&destination, b"known good bytes").unwrap();

        {
            let mut staged = AtomicFile::new(&destination).unwrap();
            staged.file_mut().write_all(b"partial replacement").unwrap();
        }

        assert_eq!(std::fs::read(&destination).unwrap(), b"known good bytes");
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    }

    #[test]
    fn noclobber_commit_preserves_existing_destination() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("artifact.bin");
        std::fs::write(&destination, b"known good bytes").unwrap();

        let error = write_bytes_noclobber(&destination, b"replacement").unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(&destination).unwrap(), b"known good bytes");
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    }
}
