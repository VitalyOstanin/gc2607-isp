//! gc2607-tune: run-time tuning window for the GC2607 camera service.
//!
//! Edits /etc/gc2607-isp/colour.conf (re-read by the service within a second),
//! shows the camera while it is free and the window is focused, and asks the
//! service for a snapshot while a call holds the camera.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use eframe::egui;
use gc2607_isp::colour::{KeyGroup, KeyKind, DEFAULT_PATH, KEYS};
use gc2607_isp::output::yuyv_to_rgba;
use gc2607_isp::presets::{valid_name, Presets};
use gc2607_isp::snapshot_request::{self, SNAPSHOT_DIR};
use gc2607_isp::tune_file::{get_value, set_value, write_atomic, WriteError};
use gc2607_isp::v4l2_capture::{CaptureError, Preview};

const DEVICE: &str = "/dev/video0";
const EXAMPLE: &str = "/usr/share/gc2607-isp/colour.conf.example";
/// At most this often while a slider is dragged (the service polls at 1 s).
const WRITE_EVERY: Duration = Duration::from_millis(200);

enum View {
    Live(Preview),
    Busy,
    Paused,
    Snapshot,
}

struct App {
    conf: PathBuf,
    text: String,
    pending: bool,
    last_write: Instant,
    status: String,
    writable: bool,
    view: View,
    texture: Option<egui::TextureHandle>,
    presets: Presets,
    preset_names: Vec<String>,
    new_preset: String,
    ab: [Option<String>; 2],
    ab_on: usize,
    paused: bool,
    on_top: bool,
    /// The text before "Default all", for its one-step undo.
    undo: Option<String>,
}

impl App {
    fn new() -> Self {
        let conf = PathBuf::from(DEFAULT_PATH);
        let text = std::fs::read_to_string(&conf).unwrap_or_default();
        let presets = Presets::user();
        let preset_names = presets.list().unwrap_or_default();
        let mut app = App {
            writable: probe_writable(&conf),
            conf,
            text,
            pending: false,
            last_write: Instant::now() - WRITE_EVERY,
            status: String::new(),
            view: View::Paused,
            texture: None,
            presets,
            preset_names,
            new_preset: String::new(),
            ab: [None, None],
            ab_on: 0,
            paused: false,
            on_top: false,
            undo: None,
        };
        if !app.writable {
            app.status = "no write access: sudo usermod -aG gc2607 $USER, then log in again".into();
        } else if app.text.is_empty() && !app.conf.exists() {
            app.status = "no colour.conf: press Reset all to create it from the example".into();
        }
        app
    }

    fn value(&self, name: &str, default: f64) -> f64 {
        get_value(&self.text, name).unwrap_or(default)
    }

    fn set(&mut self, name: &str, v: f64) {
        self.undo = None;
        self.text = set_value(&self.text, name, v);
        self.pending = true;
    }

    fn flush(&mut self, force: bool) {
        if !self.pending || !self.writable || (!force && self.last_write.elapsed() < WRITE_EVERY) {
            return;
        }
        self.pending = false;
        self.last_write = Instant::now();
        self.status = match write_atomic(&self.conf, &self.text) {
            Ok(()) => format!("written {}", self.conf.display()),
            Err(WriteError::Invalid(e)) => format!("not written: {e}"),
            Err(e) => format!("write failed: {e}"),
        };
    }

    fn replace_text(&mut self, text: String) {
        self.text = text;
        self.pending = true;
        self.flush(true);
    }

    /// Hold the camera only while the window is focused and not paused.
    fn manage_camera(&mut self, ctx: &egui::Context) {
        let focused = ctx.input(|i| {
            i.viewport().focused.unwrap_or(true) && !i.viewport().minimized.unwrap_or(false)
        });
        let want = focused && !self.paused;
        match (&self.view, want) {
            (View::Live(p), true) => {
                if let Some(err) = p.error() {
                    self.status = format!("camera: {err}");
                    self.view = View::Paused;
                }
            }
            // Dropping the Preview closes the device.
            (View::Live(_), false) => self.view = View::Paused,
            (View::Snapshot, _) => {}
            // Also retries a busy camera: the call may have ended.
            (_, true) => {
                self.view = match Preview::start(Path::new(DEVICE)) {
                    Ok(p) => View::Live(p),
                    Err(CaptureError::Busy) => View::Busy,
                    Err(CaptureError::Other(e)) => {
                        self.status = format!("camera: {e}");
                        self.paused = true;
                        View::Paused
                    }
                };
            }
            (_, false) => {}
        }
        if let View::Live(p) = &self.view {
            if let Some(f) = p.take() {
                let step = (f.width / 960).max(1);
                let rgba = yuyv_to_rgba(&f.yuyv, f.width, f.height, f.limited, step);
                let size = [f.width.div_ceil(step), f.height.div_ceil(step)];
                self.show_rgba(ctx, size, &rgba);
            }
            ctx.request_repaint_after(Duration::from_millis(33));
        } else if matches!(self.view, View::Busy) {
            ctx.request_repaint_after(Duration::from_secs(2));
        }
    }

    fn show_rgba(&mut self, ctx: &egui::Context, size: [usize; 2], rgba: &[u8]) {
        let img = egui::ColorImage::from_rgba_unmultiplied(size, rgba);
        match &mut self.texture {
            Some(t) => t.set(img, egui::TextureOptions::LINEAR),
            None => {
                self.texture = Some(ctx.load_texture("camera", img, egui::TextureOptions::LINEAR))
            }
        }
    }

    fn snapshot(&mut self, ctx: &egui::Context) {
        self.flush(true);
        match snapshot_request::request(&self.conf, Path::new(SNAPSHOT_DIR), Duration::from_secs(5))
        {
            Ok((w, h, rgba)) => {
                self.show_rgba(ctx, [w, h], &rgba);
                self.status = "snapshot from the service".into();
                if !matches!(self.view, View::Live(_)) {
                    self.view = View::Snapshot;
                }
                if let Ok(t) = std::fs::read_to_string(&self.conf) {
                    self.text = t;
                }
            }
            Err(e) => self.status = format!("snapshot failed: {e:?} (journalctl -u gc2607-camera)"),
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.add_enabled_ui(self.writable, |ui| {
            let mut group = None;
            for k in KEYS {
                if k.group == KeyGroup::Snapshot {
                    continue;
                }
                if group != Some(k.group) {
                    group = Some(k.group);
                    ui.separator();
                    ui.strong(group_title(k.group));
                }
                let cur = self.value(k.name, k.default);
                ui.horizontal(|ui| {
                    let mut v = cur;
                    match k.kind {
                        KeyKind::Switch => {
                            let mut on = v != 0.0;
                            ui.checkbox(&mut on, k.name);
                            v = if on { 1.0 } else { 0.0 };
                        }
                        KeyKind::NumberOrAuto | KeyKind::IntegerOrAuto => {
                            let mut auto = v == 0.0;
                            ui.checkbox(&mut auto, "auto");
                            if auto {
                                v = 0.0;
                                ui.label(k.name);
                            } else {
                                if v == 0.0 {
                                    v = k.min;
                                }
                                let mut s = egui::Slider::new(&mut v, k.min..=k.max).text(k.name);
                                if k.kind == KeyKind::IntegerOrAuto {
                                    s = s.integer();
                                }
                                ui.add(s);
                            }
                        }
                        KeyKind::Number | KeyKind::Integer => {
                            let mut s = egui::Slider::new(&mut v, k.min..=k.max).text(k.name);
                            if k.kind == KeyKind::Integer {
                                s = s.integer();
                            }
                            ui.add(s);
                        }
                    }
                    if ui.small_button("default").clicked() {
                        v = k.default;
                    }
                    if v != cur {
                        self.set(k.name, v);
                    }
                });
            }
        });
    }

    fn bottom_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui.button("Reset all").clicked() {
                match std::fs::read_to_string(EXAMPLE) {
                    Ok(t) => self.replace_text(t),
                    Err(e) => self.status = format!("{EXAMPLE}: {e}"),
                }
            }
            if ui
                .button("Default all")
                .on_hover_text("Every image setting to its built-in default; debug switches stay")
                .clicked()
            {
                let before = self.text.clone();
                for k in KEYS {
                    if !matches!(k.group, KeyGroup::Debug | KeyGroup::Snapshot) {
                        self.text = set_value(&self.text, k.name, k.default);
                    }
                }
                if self.text != before {
                    self.undo = Some(before);
                    self.pending = true;
                    self.flush(true);
                }
            }
            if let Some(before) = self.undo.clone() {
                if ui.button("Undo default").clicked() {
                    self.undo = None;
                    self.replace_text(before);
                }
            }
            ui.separator();
            ui.add(
                egui::TextEdit::singleline(&mut self.new_preset)
                    .hint_text("preset name")
                    .desired_width(120.0),
            );
            if ui.button("Save preset").clicked() {
                if valid_name(&self.new_preset) {
                    self.status = match self.presets.save(&self.new_preset, &self.text) {
                        Ok(()) => format!("saved preset {}", self.new_preset),
                        Err(e) => format!("preset: {e}"),
                    };
                    self.preset_names = self.presets.list().unwrap_or_default();
                } else {
                    self.status = "preset name: letters, digits, - and _ only".into();
                }
            }
            for slot in 0..2 {
                let label = ["A", "B"][slot];
                let current = self.ab[slot].clone().unwrap_or_else(|| "-".into());
                egui::ComboBox::from_label(label)
                    .selected_text(current)
                    .show_ui(ui, |ui| {
                        for n in self.preset_names.clone() {
                            let on = self.ab[slot].as_deref() == Some(n.as_str());
                            if ui.selectable_label(on, &n).clicked() {
                                self.ab[slot] = Some(n);
                            }
                        }
                    });
            }
            // Space toggles too, unless a text field has the keyboard.
            let typing = ui.ctx().egui_wants_keyboard_input();
            let toggle = ui
                .button(format!("A/B: {}", ["A", "B"][self.ab_on]))
                .clicked()
                || (!typing && ui.input(|i| i.key_pressed(egui::Key::Space)));
            if toggle {
                let next = 1 - self.ab_on;
                if let Some(name) = self.ab[next].clone() {
                    match self.presets.load(&name) {
                        Ok(t) => {
                            self.ab_on = next;
                            self.replace_text(t);
                            self.status = format!("{} = {name}", ["A", "B"][next]);
                        }
                        Err(e) => self.status = format!("preset {name}: {e}"),
                    }
                }
            }
            ui.separator();
            let pause = if self.paused {
                "Resume video"
            } else {
                "Pause video"
            };
            if ui.button(pause).clicked() {
                self.paused = !self.paused;
            }
            ui.separator();
            if ui
                .checkbox(&mut self.on_top, "On top")
                .on_hover_text("Keep this window above the call window")
                .changed()
            {
                let level = if self.on_top {
                    egui::WindowLevel::AlwaysOnTop
                } else {
                    egui::WindowLevel::Normal
                };
                ui.ctx()
                    .send_viewport_cmd(egui::ViewportCommand::WindowLevel(level));
            }
        });
    }

    fn picture(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let mut snapshot = false;
        match (&self.view, &self.texture) {
            (View::Busy, _) => {
                ui.label("Camera busy (a call holds it).");
            }
            (View::Paused, None) => {
                ui.label("Video paused.");
            }
            (_, Some(t)) => {
                ui.add(egui::Image::new(t).shrink_to_fit());
            }
            (_, None) => {
                ui.label("Waiting for the camera...");
            }
        }
        ui.horizontal(|ui| {
            if ui.button("Snapshot").clicked() {
                snapshot = true;
            }
            if matches!(self.view, View::Snapshot) && ui.button("Back to live").clicked() {
                self.view = View::Paused;
            }
        });
        if snapshot {
            self.snapshot(ctx);
        }
    }
}

fn group_title(g: KeyGroup) -> &'static str {
    match g {
        KeyGroup::WhiteBalance => "White balance",
        KeyGroup::Matrices => "Colour matrices",
        KeyGroup::Skin => "Skin band",
        KeyGroup::Yellow => "Yellow band",
        KeyGroup::Colour => "Saturation",
        KeyGroup::Exposure => "Exposure",
        KeyGroup::Awb => "Auto white balance",
        KeyGroup::Denoise => "Chroma denoise",
        KeyGroup::Debug => "Debug stages",
        KeyGroup::Snapshot => "Snapshot",
    }
}

/// Whether the directory (atomic rename) accepts new files.
fn probe_writable(conf: &Path) -> bool {
    let dir = conf.parent().unwrap_or(Path::new("."));
    let probe = dir.join(format!(".gc2607-tune-probe-{}", std::process::id()));
    let ok = std::fs::write(&probe, b"").is_ok();
    let _ = std::fs::remove_file(&probe);
    ok
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.manage_camera(&ctx);
        egui::Panel::bottom("status").show(ui, |ui| ui.label(&self.status));
        egui::Panel::bottom("actions").show(ui, |ui| self.bottom_bar(ui));
        egui::Panel::right("controls")
            .default_size(420.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
            });
        self.picture(ui, &ctx);
        let dragging = ctx.input(|i| i.pointer.any_down());
        self.flush(!dragging);
        if self.pending {
            ctx.request_repaint_after(WRITE_EVERY);
        }
    }
}

fn main() -> eframe::Result {
    // Wayland gives clients no way to keep a window on top, so run under
    // XWayland when it is there: the "On top" toggle then works.
    if std::env::var_os("DISPLAY").is_some() {
        std::env::remove_var("WAYLAND_DISPLAY");
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 800.0])
            .with_title("gc2607-tune"),
        ..Default::default()
    };
    eframe::run_native(
        "gc2607-tune",
        options,
        Box::new(|_cc| Ok(Box::new(App::new()))),
    )
}
