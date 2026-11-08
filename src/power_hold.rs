//! Hold the power-saver profile while an application captures.
//!
//! On mains power the firmware lets the package draw more, and during a call
//! the CPU and the browser spend it: the package reaches its thermal limit and
//! the integrated GPU, which shares it, stalls for tens of milliseconds, so
//! frames miss their slot. The power-saver profile keeps the package cool
//! enough for the GPU to keep pace.
//!
//! The hold goes through power-profiles-daemon: `powerprofilesctl launch`
//! takes a profile hold for as long as the command it runs is alive, and the
//! daemon drops the hold by itself when the holder leaves the bus. So the hold
//! never outlives the daemon, and a user switching the profile by hand cancels
//! it. Failures (no daemon, no `powerprofilesctl`) only cost the hold, never
//! the camera.

use std::process::{Child, Command, Stdio};

use nix::sys::signal::{killpg, Signal};
use nix::unistd::Pid;

const APP_ID: &str = "gc2607-video";
const REASON: &str = "camera in use";

pub struct PowerHold {
    command: Option<Vec<String>>,
    child: Option<Child>,
}

impl PowerHold {
    /// A hold of the power-saver profile, or a no-op when `enabled` is false.
    pub fn power_saver(enabled: bool) -> Self {
        let command = enabled.then(|| {
            [
                "powerprofilesctl",
                "launch",
                "--profile",
                "power-saver",
                "--reason",
                REASON,
                "--appid",
                APP_ID,
                "--",
                "sleep",
                "infinity",
            ]
            .map(String::from)
            .to_vec()
        });
        PowerHold {
            command,
            child: None,
        }
    }

    /// A hold that runs `command` (program and arguments) while held.
    pub fn with_command(command: Vec<String>) -> Self {
        PowerHold {
            command: Some(command),
            child: None,
        }
    }

    pub fn is_held(&self) -> bool {
        self.child.is_some()
    }

    /// Take the hold if it is not taken yet.
    pub fn acquire(&mut self) {
        let Some(cmd) = &self.command else { return };
        if self.child.is_some() {
            return;
        }
        use std::os::unix::process::CommandExt;
        // Own process group, so release stops the launcher and the command it
        // started together.
        match Command::new(&cmd[0])
            .args(&cmd[1..])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .process_group(0)
            .spawn()
        {
            Ok(child) => {
                eprintln!("power profile: holding power-saver while the camera is in use");
                self.child = Some(child);
            }
            Err(e) => eprintln!("power profile: cannot run {}: {e}", cmd[0]),
        }
    }

    /// Drop the hold if it is taken.
    pub fn release(&mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };
        if let Ok(pid) = i32::try_from(child.id()) {
            // The group is gone already if the launcher failed and exited.
            let _ = killpg(Pid::from_raw(pid), Signal::SIGTERM);
        }
        let _ = child.wait();
        eprintln!("power profile: hold released");
    }
}

impl Drop for PowerHold {
    fn drop(&mut self) {
        self.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alive(pid: u32) -> bool {
        // /proc/<pid> stays until the process is reaped; check its state too.
        std::fs::read_to_string(format!("/proc/{pid}/stat"))
            .map(|s| !s.contains(") Z "))
            .unwrap_or(false)
    }

    #[test]
    fn release_stops_the_whole_group() {
        // The shell starts a grandchild, as `powerprofilesctl launch` does.
        let mut hold = PowerHold::with_command(
            ["sh", "-c", "sleep 30 & echo $! > \"$0\"; wait"]
                .map(String::from)
                .to_vec(),
        );
        let pidfile = std::env::temp_dir().join(format!("gc2607-hold-{}", std::process::id()));
        if let Some(cmd) = hold.command.as_mut() {
            cmd.push(pidfile.display().to_string());
        }
        hold.acquire();
        assert!(hold.is_held());
        let shell = hold.child.as_ref().unwrap().id();
        let mut grandchild = None;
        for _ in 0..100 {
            if let Ok(s) = std::fs::read_to_string(&pidfile) {
                if let Ok(p) = s.trim().parse::<u32>() {
                    grandchild = Some(p);
                    break;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let grandchild = grandchild.expect("the grandchild started");
        assert!(alive(grandchild));
        hold.release();
        assert!(!hold.is_held());
        assert!(!alive(shell));
        for _ in 0..100 {
            if !alive(grandchild) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(!alive(grandchild));
        let _ = std::fs::remove_file(pidfile);
    }

    #[test]
    fn acquire_twice_keeps_one_holder() {
        let mut hold = PowerHold::with_command(["sleep", "30"].map(String::from).to_vec());
        hold.acquire();
        let first = hold.child.as_ref().unwrap().id();
        hold.acquire();
        assert_eq!(hold.child.as_ref().unwrap().id(), first);
    }

    #[test]
    fn a_missing_program_is_not_fatal() {
        let mut hold = PowerHold::with_command(vec!["gc2607-no-such-program".into()]);
        hold.acquire();
        assert!(!hold.is_held());
        hold.release();
    }

    #[test]
    fn disabled_does_nothing() {
        let mut hold = PowerHold::power_saver(false);
        hold.acquire();
        assert!(!hold.is_held());
    }
}
