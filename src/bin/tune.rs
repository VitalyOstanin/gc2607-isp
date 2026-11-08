//! gc2607-tune: run-time tuning window for the GC2607 camera service.

struct App;

impl eframe::App for App {
    fn ui(&mut self, ui: &mut eframe::egui::Ui, _frame: &mut eframe::Frame) {
        ui.label("gc2607-tune");
    }
}

fn main() -> eframe::Result {
    eframe::run_native(
        "gc2607-tune",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Ok(Box::new(App))),
    )
}
