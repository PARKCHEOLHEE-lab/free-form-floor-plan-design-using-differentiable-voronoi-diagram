//! Floor-plan frame rendering (replacing the matplotlib plot) and
//! optimization.gif assembly. Rooms are filled with the first k colors of
//! matplotlib's Accent colormap at alpha 0.5 over white, room outlines drawn
//! black, Voronoi cell edges gray, and sites as black dots — mirroring
//! `FloorPlanGenerator.plot`.

use crate::loss::{rooms_group, union_group};
use geo::{BoundingRect, Polygon};
use std::fs::File;
use std::io;
use std::path::Path;
use tiny_skia::{Color, FillRule, Paint, PathBuilder, Pixmap, Stroke, Transform};

pub const FRAME_SIZE: u32 = 800;

/// First 8 colors of matplotlib's Accent colormap.
const ACCENT: [[u8; 3]; 8] = [
    [127, 201, 127],
    [190, 174, 212],
    [253, 192, 134],
    [255, 255, 153],
    [56, 108, 176],
    [240, 2, 127],
    [191, 91, 23],
    [102, 102, 102],
];

struct WorldToPixel {
    min_x: f64,
    min_y: f64,
    scale: f64,
    size: f64,
}

impl WorldToPixel {
    fn new(boundary: &Polygon<f64>) -> Self {
        let rect = boundary.bounding_rect().expect("nonempty boundary");
        let span = rect.width().max(rect.height());
        let size = FRAME_SIZE as f64;
        // 5% margin on each side
        let scale = size * 0.9 / span;
        WorldToPixel {
            min_x: rect.min().x - (span - rect.width()) / 2.0,
            min_y: rect.min().y - (span - rect.height()) / 2.0,
            scale,
            size,
        }
    }

    fn map(&self, x: f64, y: f64) -> (f32, f32) {
        let px = (x - self.min_x) * self.scale + self.size * 0.05;
        // flip y: world up = screen up
        let py = self.size - ((y - self.min_y) * self.scale + self.size * 0.05);
        (px as f32, py as f32)
    }
}

fn ring_path(ring: &geo::LineString<f64>, t: &WorldToPixel) -> Option<tiny_skia::Path> {
    let pts = &ring.0;
    if pts.len() < 4 {
        return None;
    }
    let mut pb = PathBuilder::new();
    let (x0, y0) = t.map(pts[0].x, pts[0].y);
    pb.move_to(x0, y0);
    for c in &pts[1..pts.len() - 1] {
        let (x, y) = t.map(c.x, c.y);
        pb.line_to(x, y);
    }
    pb.close();
    pb.finish()
}

fn polygon_path(poly: &Polygon<f64>, t: &WorldToPixel) -> Option<tiny_skia::Path> {
    let mut pb = PathBuilder::new();
    for ring in std::iter::once(poly.exterior()).chain(poly.interiors()) {
        let pts = &ring.0;
        if pts.len() < 4 {
            continue;
        }
        let (x0, y0) = t.map(pts[0].x, pts[0].y);
        pb.move_to(x0, y0);
        for c in &pts[1..pts.len() - 1] {
            let (x, y) = t.map(c.x, c.y);
            pb.line_to(x, y);
        }
        pb.close();
    }
    pb.finish()
}

/// Renders one frame as straight RGBA8888 bytes (FRAME_SIZE x FRAME_SIZE).
pub fn render_frame(
    boundary: &Polygon<f64>,
    cells_sorted: &[Polygon<f64>],
    room_indices: &[usize],
    n_rooms: usize,
    sites: &[[f32; 2]],
) -> Vec<u8> {
    let t = WorldToPixel::new(boundary);
    let mut pixmap = Pixmap::new(FRAME_SIZE, FRAME_SIZE).expect("pixmap");
    pixmap.fill(Color::WHITE);

    let groups = rooms_group(cells_sorted, room_indices);
    let mut paint = Paint::default();
    paint.anti_alias = true;

    // room fills (alpha 0.5) + black outlines
    let mut stroke = Stroke::default();
    stroke.width = 2.0;
    for (gi, group) in groups.iter().enumerate().take(n_rooms.max(groups.len())) {
        let union = union_group(group);
        let rgb = ACCENT[gi % ACCENT.len()];
        for piece in &union {
            if let Some(path) = polygon_path(piece, &t) {
                paint.set_color(Color::from_rgba8(rgb[0], rgb[1], rgb[2], 128));
                pixmap.fill_path(&path, &paint, FillRule::EvenOdd, Transform::identity(), None);
                paint.set_color(Color::BLACK);
                pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
            }
        }
    }

    // voronoi cell edges, thin gray
    let mut thin = Stroke::default();
    thin.width = 0.8;
    paint.set_color(Color::from_rgba8(128, 128, 128, 255));
    for cell in cells_sorted {
        if let Some(path) = ring_path(cell.exterior(), &t) {
            pixmap.stroke_path(&path, &paint, &thin, Transform::identity(), None);
        }
    }

    // sites as black dots
    paint.set_color(Color::BLACK);
    for s in sites {
        let (x, y) = t.map(s[0] as f64, s[1] as f64);
        if let Some(circle) = PathBuilder::from_circle(x, y, 3.0) {
            pixmap.fill_path(&circle, &paint, FillRule::Winding, Transform::identity(), None);
        }
    }

    // tiny-skia stores premultiplied RGBA; everything here is opaque over
    // white, so demultiply() yields exact straight RGBA
    pixmap
        .pixels()
        .iter()
        .flat_map(|p| {
            let d = p.demultiply();
            [d.red(), d.green(), d.blue(), d.alpha()]
        })
        .collect()
}

/// Streams frames into optimization.gif (20ms per frame, like the Python
/// `duration=20`) without holding the whole animation in memory.
pub struct GifWriter {
    encoder: gif::Encoder<File>,
}

impl GifWriter {
    pub fn create(path: &Path) -> io::Result<Self> {
        let file = File::create(path)?;
        let size = FRAME_SIZE as u16;
        let mut encoder = gif::Encoder::new(file, size, size, &[])
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        encoder
            .set_repeat(gif::Repeat::Infinite)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        Ok(GifWriter { encoder })
    }

    pub fn add_frame(&mut self, frame_rgba: &[u8]) -> io::Result<()> {
        let size = FRAME_SIZE as u16;
        let mut data = frame_rgba.to_vec();
        let mut frame = gif::Frame::from_rgba_speed(size, size, &mut data, 10);
        frame.delay = 2; // 20ms in 1/100s units
        self.encoder
            .write_frame(&frame)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
    }
}

/// Assembles RGBA frames into optimization.gif in one call.
pub fn save_gif(frames: &[Vec<u8>], path: &Path) -> io::Result<()> {
    let mut writer = GifWriter::create(path)?;
    for frame in frames {
        writer.add_frame(frame)?;
    }
    Ok(())
}
