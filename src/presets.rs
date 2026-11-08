//! Named copies of the tuning file for A/B comparison, kept under
//! `$XDG_CONFIG_HOME/gc2607-tune/presets/<name>.conf`.

use std::io;
use std::path::PathBuf;

/// A preset name: 1..=64 of `[A-Za-z0-9_-]`, so it can never leave its
/// directory or hide as a dot file.
pub fn valid_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// A directory of presets, one `<name>.conf` file each.
pub struct Presets {
    dir: PathBuf,
}

impl Presets {
    /// The per-user preset directory. A relative `XDG_CONFIG_HOME` is ignored,
    /// as the XDG base directory specification asks.
    pub fn user() -> Self {
        let base = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
            .unwrap_or_else(|| PathBuf::from("."));
        Presets::at(base.join("gc2607-tune").join("presets"))
    }

    /// Presets kept in `dir`.
    pub fn at(dir: PathBuf) -> Self {
        Presets { dir }
    }

    fn path(&self, name: &str) -> io::Result<PathBuf> {
        if !valid_name(name) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("bad preset name {name:?}"),
            ));
        }
        Ok(self.dir.join(format!("{name}.conf")))
    }

    /// Preset names, sorted. A missing directory is an empty list.
    pub fn list(&self) -> io::Result<Vec<String>> {
        let mut names = Vec::new();
        match std::fs::read_dir(&self.dir) {
            Ok(rd) => {
                for e in rd {
                    let p = e?.path();
                    if p.extension().and_then(|x| x.to_str()) != Some("conf") {
                        continue;
                    }
                    if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                        if valid_name(stem) {
                            names.push(stem.to_string());
                        }
                    }
                }
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        names.sort();
        Ok(names)
    }

    /// The text of preset `name`.
    pub fn load(&self, name: &str) -> io::Result<String> {
        std::fs::read_to_string(self.path(name)?)
    }

    /// Store `text` as preset `name`, replacing one of the same name.
    pub fn save(&self, name: &str, text: &str) -> io::Result<()> {
        let path = self.path(name)?;
        std::fs::create_dir_all(&self.dir)?;
        std::fs::write(path, text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_plain() {
        assert!(valid_name("variant-10_day"));
        let long = "x".repeat(65);
        for bad in ["", "../x", "a/b", ".hidden", "a b", long.as_str()] {
            assert!(!valid_name(bad), "{bad:?}");
        }
    }

    #[test]
    fn save_list_load() {
        let dir = std::env::temp_dir().join(format!("gc2607-presets-{}", std::process::id()));
        let store = Presets::at(dir.clone());
        assert!(store.list().unwrap().is_empty());
        store.save("b", "saturation = 1.1\n").unwrap();
        store.save("a", "saturation = 1.2\n").unwrap();
        std::fs::write(dir.join("notes.txt"), "not a preset").unwrap();
        assert_eq!(
            store.list().unwrap(),
            vec!["a".to_string(), "b".to_string()]
        );
        assert_eq!(store.load("a").unwrap(), "saturation = 1.2\n");
        assert!(store.save("../evil", "x").is_err());
        assert!(store.load("../evil").is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
