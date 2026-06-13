//! Minimal tensorboard event-file writer (scalars only), replacing
//! `torch.utils.tensorboard.SummaryWriter` for the 7 loss tags. Each record
//! is TFRecord-framed (length + masked CRC32C + Event protobuf payload +
//! masked CRC32C), hand-encoded — the Event/Summary/Value subset needed for
//! scalars is tiny and stable.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

pub struct TfEventsWriter {
    out: BufWriter<File>,
}

fn masked_crc(data: &[u8]) -> u32 {
    // TFRecord's masked CRC: rotate_right(15) then offset, per the TensorFlow
    // record format.
    let c = crc32c::crc32c(data);
    c.rotate_right(15).wrapping_add(0xa282ead8)
}

fn put_varint(buf: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            buf.push(b);
            break;
        }
        buf.push(b | 0x80);
    }
}

/// Event { double wall_time = 1; int64 step = 2; string file_version = 3;
///         Summary summary = 5; }
/// Summary { repeated Value value = 1; }
/// Value { string tag = 1; float simple_value = 2; }
fn encode_event(
    wall_time: f64,
    step: Option<i64>,
    file_version: Option<&str>,
    scalar: Option<(&str, f32)>,
) -> Vec<u8> {
    let mut ev = Vec::with_capacity(64);
    ev.push(0x09); // field 1, 64-bit
    ev.extend_from_slice(&wall_time.to_le_bytes());
    if let Some(s) = step {
        ev.push(0x10); // field 2, varint
        put_varint(&mut ev, s as u64);
    }
    if let Some(fv) = file_version {
        ev.push(0x1A); // field 3, length-delimited
        put_varint(&mut ev, fv.len() as u64);
        ev.extend_from_slice(fv.as_bytes());
    }
    if let Some((tag, value)) = scalar {
        let mut val = Vec::with_capacity(tag.len() + 8);
        val.push(0x0A); // Value.tag
        put_varint(&mut val, tag.len() as u64);
        val.extend_from_slice(tag.as_bytes());
        val.push(0x15); // Value.simple_value, 32-bit
        val.extend_from_slice(&value.to_le_bytes());

        let mut summary = Vec::with_capacity(val.len() + 4);
        summary.push(0x0A); // Summary.value
        put_varint(&mut summary, val.len() as u64);
        summary.extend_from_slice(&val);

        ev.push(0x2A); // Event.summary
        put_varint(&mut ev, summary.len() as u64);
        ev.extend_from_slice(&summary);
    }
    ev
}

impl TfEventsWriter {
    pub fn create(log_dir: &Path, wall_time: f64) -> io::Result<Self> {
        std::fs::create_dir_all(log_dir)?;
        let name = format!("events.out.tfevents.{}.rust", wall_time as u64);
        let file = File::create(log_dir.join(name))?;
        let mut writer = TfEventsWriter {
            out: BufWriter::new(file),
        };
        writer.write_record(&encode_event(wall_time, None, Some("brain.Event:2"), None))?;
        Ok(writer)
    }

    pub fn add_scalar(&mut self, tag: &str, value: f32, step: i64, wall_time: f64) -> io::Result<()> {
        self.write_record(&encode_event(wall_time, Some(step), None, Some((tag, value))))
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }

    fn write_record(&mut self, payload: &[u8]) -> io::Result<()> {
        let len = (payload.len() as u64).to_le_bytes();
        self.out.write_all(&len)?;
        self.out.write_all(&masked_crc(&len).to_le_bytes())?;
        self.out.write_all(payload)?;
        self.out.write_all(&masked_crc(payload).to_le_bytes())?;
        Ok(())
    }
}
