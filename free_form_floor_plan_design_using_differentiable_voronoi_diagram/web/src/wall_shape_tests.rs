use super::*;
use voronoi_floorplan::loss::WallLossMode;

#[test]
fn explicit_browser_wall_mode_reports_angle_and_step_in_preview_and_optimization(){
    let mut opt=WasmOpt::build(shapes::by_name("shape_a").unwrap().polygon(),18,vec![0.5,0.3,0.2],LossWeights{w_wall:1.,..Default::default()},777,0.005);
    let read=|opt:&WasmOpt|loss::floor_plan_loss_with_program(&opt.sites,&opt.boundary,&opt.target_areas,&opt.room_indices,&opt.weights,None,opt.program.as_ref());
    let legacy=read(&opt);
    opt.set_wall_loss(0.35).unwrap();
    assert!(matches!(opt.weights.wall_mode,WallLossMode::CosineSquared),"browser configuration must explicitly replace the legacy direction term");
    let current=read(&opt);let preview=opt.geometry_frame(&current);
    assert!(current.wall>0. && current.wall<legacy.wall);
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
    assert!(matches!(opt.weights.wall_mode,WallLossMode::CosineSquared));
}
