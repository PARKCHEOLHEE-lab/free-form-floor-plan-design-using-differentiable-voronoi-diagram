use super::*;
use voronoi_floorplan::loss::WallLossMode;

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
fn step_weight_keeps_legacy_walls_in_preview_and_optimization(){
    let mut opt=WasmOpt::build(shapes::by_name("shape_a").unwrap().polygon(),18,vec![0.5,0.3,0.2],LossWeights{w_wall:1.,..Default::default()},777,0.005);
    let read=|opt:&WasmOpt|loss::floor_plan_loss_with_program(&opt.sites,&opt.boundary,&opt.target_areas,&opt.room_indices,&opt.weights,None,opt.program.as_ref());
    let legacy=read(&opt);
    opt.set_wall_loss(0.35).unwrap();
    assert!(matches!(opt.weights.wall_mode,WallLossMode::LegacyL1),"changing the step weight keeps the previous wall model");
    let current=read(&opt);let preview=opt.geometry_frame(&current);
    assert!(current.wall>0.);
    assert_eq!(current.wall,legacy.wall,"step and wall are separate costs");
    assert!(preview.jog>0.,"preview must serialize the active step contribution");
    assert!((preview.loss-preview.wall-preview.jog).abs()<1e-5);
    let step=opt.advance();
    assert!((step.wall-preview.wall).abs()<1e-4 && (step.jog-preview.jog).abs()<1e-4,"preview and actual local objective use the same model");
    for bad in [-1.,f64::NAN,f64::INFINITY]{
        assert!(opt.configure_wall_loss(bad).is_err());
        assert_eq!(opt.weights.w_jog,0.35,"invalid configuration preserves the valid weights");
    }
    opt.configure_wall_loss(0.).unwrap();
    assert_eq!(read(&opt).jog,0.);
    assert!(matches!(opt.weights.wall_mode,WallLossMode::LegacyL1));
}
