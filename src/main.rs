//! Offline CLI: process one raw SGRBG10 frame into an image.
//!
//! Usage: gc2607-isp [--half|--mhc] <input.bin> [output]
//!   --mhc   full-resolution Malvar-He-Cutler debayer (default)
//!   --half  half-resolution debayer (matches the golden reference)
//! Output format is chosen by extension: `.png` is written via the image
//! crate, anything else is written as binary PPM (P6). Prints the scene
//! estimate (gains, CCT, LSC source).

use std::io::{self, Write};
use std::process::ExitCode;

use gc2607_isp::pipeline::{self, DebayerMode};
use gc2607_isp::raw;

fn main() -> ExitCode {
    let mut mode = DebayerMode::Mhc;
    let mut positional: Vec<String> = Vec::new();
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--half" => mode = DebayerMode::HalfRes,
            "--mhc" => mode = DebayerMode::Mhc,
            _ => positional.push(arg),
        }
    }
    if positional.is_empty() {
        eprintln!("usage: gc2607-isp [--half|--mhc] <input.bin> [output.png|output.ppm]");
        return ExitCode::FAILURE;
    }
    let input = &positional[0];
    let output = positional.get(1).map(String::as_str).unwrap_or("out.png");

    let frame = match raw::load_blc(input) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error reading raw '{input}': {e}");
            return ExitCode::FAILURE;
        }
    };

    let (w, h, rgb, est) = pipeline::process_with(&frame, mode);

    let write_result = if output.ends_with(".png") {
        write_png(output, w, h, &rgb)
    } else {
        write_ppm(output, w, h, &rgb)
    };
    if let Err(e) = write_result {
        eprintln!("error writing '{output}': {e}");
        return ExitCode::FAILURE;
    }

    println!(
        "{w}x{h}  mode={:?}  chroma r/g={:.3} b/g={:.3}  gains R/B={:.3}/{:.3}  CCT={:.0}K  LSC LS{}",
        mode, est.chroma.0, est.chroma.1, est.gains[0], est.gains[2], est.cct, est.ls
    );
    println!("-> {output}");
    ExitCode::SUCCESS
}

fn write_ppm(path: &str, w: usize, h: usize, rgb: &[u8]) -> io::Result<()> {
    let f = std::fs::File::create(path)?;
    let mut bw = io::BufWriter::new(f);
    write!(bw, "P6\n{w} {h}\n255\n")?;
    bw.write_all(rgb)?;
    bw.flush()
}

fn write_png(path: &str, w: usize, h: usize, rgb: &[u8]) -> io::Result<()> {
    let buf: image::RgbImage = image::ImageBuffer::from_raw(w as u32, h as u32, rgb.to_vec())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "buffer size mismatch"))?;
    buf.save(path).map_err(|e| io::Error::other(e.to_string()))
}
