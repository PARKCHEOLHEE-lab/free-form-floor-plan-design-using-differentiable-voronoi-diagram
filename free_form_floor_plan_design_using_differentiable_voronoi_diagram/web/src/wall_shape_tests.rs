use super::*;

#[test]
fn preset_area_fit_leaves_all_sites_free_to_move_and_is_safe_to_retry() {
    let boundary = shapes::by_name("shape_e").unwrap().polygon();
    let mut opt = WasmOpt::build(
        boundary, 80, vec![0.13, 0.11, 0.21, 0.20, 0.11, 0.24],
        LossWeights { w_area: 0.3, w_lloyd: 0.01, ..Default::default() }, 4971, 0.005,
    );
    let sites = opt.sites.clone();
    opt.configure_preset_areas().unwrap();
    let fitted = opt.room_indices.clone();
    opt.configure_preset_areas().unwrap();
    assert_eq!(opt.room_indices, fitted);
    assert_eq!(opt.sites, sites, "preset fitting preserves existing site positions and indices");
    assert!(opt.program.is_none(), "preset fitting does not add room center anchors");
    opt.advance();
    assert!(opt.configure_preset_areas().is_err());
    assert_ne!(&opt.sites[..6], &sites[..6], "sites used to initialize rooms can still move");
    assert_eq!(opt.room_indices, fitted, "a rejected refit leaves the running plan intact");
}

#[test]
fn cosine_weight_keeps_the_l1_weight() {
    let mut opt = WasmOpt::from_shape(
        "shape_e", 24, &[0.5, 0.3, 0.2], 1. / 13., 0., 0., 0., 0., 0., 4971, 0.005,
    ).unwrap();
    opt.set_cosine_weight(1.).unwrap();
    for bad in [-1., f64::NAN, f64::INFINITY] {
        assert!(opt.configure_cosine_weight(bad).is_err());
    }
    assert_eq!(opt.weights.w_cosine, 1., "invalid configuration preserves the valid weight");
    assert_eq!(opt.weights.w_wall, 1. / 13.);
    let frame = opt.advance();
    assert!(frame.wall > 0. && frame.cosine.unwrap() > 0.);
    assert!((frame.loss - frame.wall - frame.cosine.unwrap()).abs() < 1e-5, "the total is L1 plus Cos, with no other wall term");
    opt.set_cosine_weight(0.).unwrap();
    assert!(opt.advance().cosine.is_none());
}
