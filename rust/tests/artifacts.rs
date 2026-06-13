//! KR8: artifact tests — the tfevents writer produces TFRecord-framed,
//! CRC-valid scalar events that round-trip through a minimal parser, and the
//! GIF renderer produces decodable, non-blank frames.

use voronoi_floorplan::{render, shapes, tfevents::TfEventsWriter, voronoi};

fn unmask_crc(masked: u32) -> u32 {
    // inverse of masked_crc: subtract the offset, then rotate_left(15)
    masked.wrapping_sub(0xa282ead8).rotate_left(15)
}

/// Minimal TFRecord + Event parser: returns (file_version_seen, scalars).
fn parse_tfevents(data: &[u8]) -> (bool, Vec<(String, i64, f32)>) {
    let mut pos = 0;
    let mut file_version = false;
    let mut scalars = vec![];
    while pos + 12 <= data.len() {
        let len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
        let len_crc = u32::from_le_bytes(data[pos + 8..pos + 12].try_into().unwrap());
        assert_eq!(
            unmask_crc(len_crc),
            crc32c::crc32c(&data[pos..pos + 8]),
            "length CRC mismatch"
        );
        pos += 12;
        let payload = &data[pos..pos + len];
        let payload_crc =
            u32::from_le_bytes(data[pos + len..pos + len + 4].try_into().unwrap());
        assert_eq!(
            unmask_crc(payload_crc),
            crc32c::crc32c(payload),
            "payload CRC mismatch"
        );
        pos += len + 4;

        // parse Event: we care about field 2 (step), 3 (file_version), 5 (summary)
        let (mut p, mut step, mut summary): (usize, i64, Option<&[u8]>) = (0, 0, None);
        while p < payload.len() {
            let key = payload[p];
            p += 1;
            match key {
                0x09 => p += 8, // wall_time double
                0x10 => {
                    let (v, np) = read_varint(payload, p);
                    step = v as i64;
                    p = np;
                }
                0x1A => {
                    let (l, np) = read_varint(payload, p);
                    p = np + l as usize;
                    file_version = true;
                }
                0x2A => {
                    let (l, np) = read_varint(payload, p);
                    summary = Some(&payload[np..np + l as usize]);
                    p = np + l as usize;
                }
                _ => panic!("unexpected Event field key {key:#x}"),
            }
        }
        if let Some(s) = summary {
            // Summary { repeated Value value = 1 } ; Value { tag=1, simple_value=2 }
            let mut q = 0;
            while q < s.len() {
                assert_eq!(s[q], 0x0A);
                let (l, nq) = read_varint(s, q + 1);
                let val = &s[nq..nq + l as usize];
                q = nq + l as usize;
                let mut r = 0;
                let mut tag = String::new();
                let mut sv = 0.0f32;
                while r < val.len() {
                    match val[r] {
                        0x0A => {
                            let (tl, nr) = read_varint(val, r + 1);
                            tag = String::from_utf8(val[nr..nr + tl as usize].to_vec()).unwrap();
                            r = nr + tl as usize;
                        }
                        0x15 => {
                            sv = f32::from_le_bytes(val[r + 1..r + 5].try_into().unwrap());
                            r += 5;
                        }
                        k => panic!("unexpected Value field key {k:#x}"),
                    }
                }
                scalars.push((tag, step, sv));
            }
        }
    }
    assert_eq!(pos, data.len(), "trailing bytes in tfevents file");
    (file_version, scalars)
}

fn read_varint(buf: &[u8], mut pos: usize) -> (u64, usize) {
    let mut out = 0u64;
    let mut shift = 0;
    loop {
        let b = buf[pos];
        pos += 1;
        out |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    (out, pos)
}

#[test]
fn tfevents_writer_round_trips_seven_scalar_tags() {
    let dir = std::env::temp_dir().join("voronoi_floorplan_tfevents_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let tags = ["loss", "loss_wall", "loss_area", "loss_lloyd", "loss_topo", "loss_bb", "loss_cell_area"];
    let mut w = TfEventsWriter::create(&dir, 1718180000.0).unwrap();
    for step in 1..=2i64 {
        for (i, tag) in tags.iter().enumerate() {
            w.add_scalar(tag, (step as f32) * 10.0 + i as f32, step, 1718180000.5).unwrap();
        }
    }
    w.flush().unwrap();

    let file = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(Result::ok)
        .find(|e| e.file_name().to_string_lossy().starts_with("events.out.tfevents"))
        .expect("an events.out.tfevents* file must exist");
    let data = std::fs::read(file.path()).unwrap();
    let (file_version, scalars) = parse_tfevents(&data);

    assert!(file_version, "first record must carry the file_version header");
    assert_eq!(scalars.len(), 14, "7 tags x 2 steps");
    for step in 1..=2i64 {
        for (i, tag) in tags.iter().enumerate() {
            let expected = (step as f32) * 10.0 + i as f32;
            assert!(
                scalars
                    .iter()
                    .any(|(t, s, v)| t == tag && *s == step && *v == expected),
                "missing scalar ({tag}, {step}, {expected})"
            );
        }
    }
}

#[test]
fn gif_renderer_produces_decodable_nonblank_frames() {
    let data = std::fs::read_to_string("fixtures/shape_a.checkpoint.json").unwrap();
    let v: serde_json::Value = serde_json::from_str(&data).unwrap();
    let sites: Vec<[f32; 2]> = serde_json::from_value(v["initial_sites"].clone()).unwrap();
    let rooms: Vec<usize> = serde_json::from_value(v["room_indices"].clone()).unwrap();
    let boundary = shapes::by_name("shape_a").unwrap().polygon();
    let geom = voronoi::compute_cells(&sites, &boundary, None);

    let frame = render::render_frame(&boundary, &geom.cells_sorted, &rooms, 4, &sites);
    assert_eq!(
        frame.len(),
        (render::FRAME_SIZE * render::FRAME_SIZE * 4) as usize,
        "frame must be RGBA at FRAME_SIZE^2"
    );
    let distinct: std::collections::HashSet<&[u8]> = frame.chunks_exact(4).collect();
    assert!(
        distinct.len() >= 4,
        "frame must contain at least 4 distinct colors (rooms + lines), got {}",
        distinct.len()
    );

    let dir = std::env::temp_dir().join("voronoi_floorplan_gif_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let gif_path = dir.join("optimization.gif");
    render::save_gif(&[frame.clone(), frame], &gif_path).unwrap();

    let mut opts = gif::DecodeOptions::new();
    opts.set_color_output(gif::ColorOutput::RGBA);
    let mut decoder = opts.read_info(std::fs::File::open(&gif_path).unwrap()).unwrap();
    let mut n_frames = 0;
    while decoder.read_next_frame().unwrap().is_some() {
        n_frames += 1;
    }
    assert_eq!(n_frames, 2, "gif must contain both frames");
    assert_eq!(decoder.width() as u32, render::FRAME_SIZE);
    assert_eq!(decoder.height() as u32, render::FRAME_SIZE);
}
