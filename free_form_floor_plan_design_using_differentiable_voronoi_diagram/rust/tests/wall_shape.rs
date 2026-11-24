use geo::{LineString,Polygon};
use voronoi_floorplan::{grad_local::LocalGradContext,loss::{floor_plan_loss_with_program,LossWeights},program::RoomProgram};

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

#[test]
fn l1_and_cosine_have_independent_weights() {
    let (boundary, mut sites, labels, _) = fixture();
    sites[5][0] += 0.015;
    let targets = [2., 2.];
    let evaluate = |l1, cosine| {
        let weights = LossWeights { w_wall: l1, w_cosine: cosine, ..Default::default() };
        let full = floor_plan_loss_with_program(&sites, &boundary, &targets, &labels, &weights, None, None);
        let local = LocalGradContext::new(&sites, &boundary, &targets, &labels, &weights).base_breakdown();
        assert!((local.wall - full.wall).abs() < 1e-5);
        assert_eq!(local.cosine, full.cosine);
        full
    };
    let l1 = evaluate(1. / 13., 0.);
    let cosine = evaluate(0., 1.);
    let mixed = evaluate(1. / 13., 1.);
    assert!(l1.wall > 0. && cosine.cosine > 0., "both costs must contribute independently");
    assert_eq!(l1.cosine, 0.);
    assert_eq!(cosine.wall, 0.);
    assert_eq!(mixed.wall, l1.wall);
    assert_eq!(mixed.cosine, cosine.cosine);
    assert_eq!(mixed.total, mixed.wall + mixed.cosine);
}

#[test]
fn cosine_term_drives_global_and_local_optimization(){
    let (b,mut sites,labels,_)=fixture();let targets=[2.,2.];
    sites[5][0]+=0.015;
    let cosine=LossWeights{w_cosine:1.,..Default::default()};
    let global=floor_plan_loss_with_program(&sites,&b,&targets,&labels,&cosine,None,None);
    assert!(global.cosine>0.,"Cos alone must enter the real global objective");
    assert_eq!(global.total,global.cosine);
    let ctx=LocalGradContext::new(&sites,&b,&targets,&labels,&cosine);
    assert!((ctx.base_total()-global.total).abs()<1e-5);
    let grads=ctx.gradients(&sites);
    assert!(grads.iter().flatten().all(|g|g.is_finite()));
    assert!(grads.iter().flatten().any(|g|g.abs()>0.01),"Cos must move sites");
    for (i,axis) in [(3,0),(5,0),(6,1)] {
        let mut moved=sites.clone();moved[i][axis]+=1e-4;
        let full=floor_plan_loss_with_program(&moved,&b,&targets,&labels,&cosine,None,None);
        assert!((ctx.local_total(&moved,i)-full.total).abs()<1e-4,"local cache must rescore changed walls: site={i}, local={}, global={}",ctx.local_total(&moved,i),full.total);
    }
}
