use geo::{Area,Contains,LineString,MultiPolygon,Point,Polygon};
use voronoi_floorplan::{grad_local::LocalGradContext,loss::{floor_plan_loss_with_program,LossWeights,WallLossMode},optim::AdamW,program::RoomProgram,voronoi};

fn fixture()->(Polygon<f64>,Vec<[f32;2]>,Vec<usize>,RoomProgram){
    let boundary=Polygon::new(LineString::from(vec![(-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.),(-1.,-1.)]),vec![]);
    let mut sites=vec![[-0.12,-0.75],[0.12,-0.75]];let mut labels=vec![0,1];
    for y in [-0.75_f32,-0.25,0.25,0.75] {for x in [-0.12,0.,0.12] {
        if sites.contains(&[x,y]) {continue;}
        sites.push([x,y]);labels.push(if x<0. || (x==0. && y.abs()<0.5) {0}else{1});
    }}
    let mut program=RoomProgram::new(&boundary,&[[-0.12,-0.75,0.03],[0.12,-0.75,0.03]]).unwrap();
    program.set_connections(&[[0,1]],0.1).unwrap();
    (boundary,sites,labels,program)
}
fn rooms(sites:&[[f32;2]],labels:&[usize],boundary:&Polygon<f64>)->Vec<MultiPolygon<f64>>{
    let cells=voronoi::render_cells(sites,boundary);
    (0..2).map(|r|voronoi_floorplan::loss::union_group(&cells.iter().filter(|(i,_)|labels[*i]==r).map(|(_,p)|p).collect::<Vec<_>>())).collect()
}
// Measure the actual room boundary at three fixed heights, independent of the loss.
fn step_span(rooms:&[MultiPolygon<f64>])->f64{
    let xs:Vec<f64>=[-0.7,0.,0.7].iter().map(|&y|rooms[0].iter().flat_map(|p|p.exterior().lines())
        .filter(|l|(l.start.y<=y&&l.end.y>y)||(l.end.y<=y&&l.start.y>y))
        .map(|l|l.start.x+(y-l.start.y)*(l.end.x-l.start.x)/(l.end.y-l.start.y))
        .fold(f64::NEG_INFINITY,f64::max)).collect();
    xs.iter().copied().fold(f64::NEG_INFINITY,f64::max)-xs.iter().copied().fold(f64::INFINITY,f64::min)
}
#[test]
fn wall_shape_terms_drive_global_local_and_feasible_optimization(){
    let (b,sites,labels,_program)=fixture();let targets=[2.,2.];
    let jog_only=LossWeights{w_jog:1.,wall_mode:WallLossMode::CosineSquared,..Default::default()};
    let global=floor_plan_loss_with_program(&sites,&b,&targets,&labels,&jog_only,None,None);
    assert!(global.jog>0.,"jog alone must enter the real global objective");
    assert_eq!(global.total,global.jog);
    let ctx=LocalGradContext::new(&sites,&b,&targets,&labels,&jog_only);
    assert!((ctx.base_total()-global.total).abs()<1e-5);
    let grads=ctx.gradients(&sites);
    assert!(grads.iter().flatten().all(|g|g.is_finite()));
    assert!(grads.iter().flatten().any(|g|g.abs()>0.01),"jog must move sites");
    for (i,axis) in [(3,0),(6,1)] {
        let mut moved=sites.clone();moved[i][axis]+=1e-4;
        let full=floor_plan_loss_with_program(&moved,&b,&targets,&labels,&jog_only,None,None);
        assert!((ctx.local_total(&moved,i)-full.total).abs()<1e-4,"local cache must rebuild jog unions: site={i}, local={}, global={}",ctx.local_total(&moved,i),full.total);
    }
    // The same mode is used without a program and when only angle is enabled.
    let mut tilted=sites.clone();tilted[5][0]+=0.015;
    let angle=LossWeights{w_wall:1.,wall_mode:WallLossMode::CosineSquared,..Default::default()};
    let g=floor_plan_loss_with_program(&tilted,&b,&targets,&labels,&angle,None,None);
    let l=LocalGradContext::new(&tilted,&b,&targets,&labels,&angle);
    assert!(g.wall>0. && g.wall<1. && g.jog==0.);
    assert!((l.base_total()-g.total).abs()<1e-4);
    assert!(l.gradients(&tilted).iter().flatten().any(|g|g.abs()>0.01));
}

#[test]
fn wall_shape_optimizer_reduces_step_displacement_with_feasible_rooms(){
    let (b,sites,labels,program)=fixture();let targets=[2.,2.];
    let run=|jog|{
        let w=LossWeights{w_wall:0.5,w_jog:jog,w_area:5.,w_lloyd:0.001,w_topo:1.,wall_mode:WallLossMode::CosineSquared,..Default::default()};
        let mut result=sites.clone();let mut opt=AdamW::new(result.len()-2,0.002);
        // Browser #iters and presets use 800 steps (web/index.html).
        for _ in 0..800 {
            let ctx=LocalGradContext::new_with_program(&result,&b,&targets,&labels,&w,Some(&program));
            let grads=ctx.gradients(&result);opt.step(&mut result[2..],&grads[2..]);
        }
        result
    };
    let before=step_span(&rooms(&sites,&labels,&b));
    let enabled=run(2.);let disabled=run(0.);
    let final_rooms=rooms(&enabled,&labels,&b);let after=step_span(&final_rooms);let control=step_span(&rooms(&disabled,&labels,&b));
    eprintln!("raw step span before={before}, after={after}, disabled={control}");
    assert!(after<before*0.5,"actual step displacement must decrease at least 50%");
    assert!(after<control*0.75,"jog must improve on the same angle-only optimization");
    assert_eq!(&enabled[..2],&sites[..2]);
    assert!(program.connection_loss(&final_rooms)<1e-4,"requested door still fits");
    for (r,room) in final_rooms.iter().enumerate(){
        assert_eq!(room.0.len(),1,"room stays connected");
        assert!(room.contains(&Point::new(sites[r][0] as f64,sites[r][1] as f64)));
        assert!((room.unsigned_area()/b.unsigned_area()-0.5).abs()<0.05,"target area remains within 5pp");
    }
}
