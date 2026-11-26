use super::*;
use geo::{LineString,MultiPolygon};

#[test]
fn cosine_edit_reuses_unchanged_pairs_and_scores_only_the_changed_one() {
    let boundary = rectangle();
    let scene = |offset| {
        let rooms = [
            room(vec![[-2.,-1.],[-1.,-1.],[-1.,1.],[-2.,1.],[-2.,-1.]]),
            room(vec![[-1.,-1.],[0.,-1.],[0.,0.],[offset,0.2],[offset,1.],[-1.,1.],[-1.,-1.]]),
            room(vec![[0.,-1.],[0.7,-1.],[0.7,1.],[offset,1.],[offset,0.2],[0.,0.],[0.,-1.]]),
            room(vec![[0.7,-1.],[1.3,-1.],[1.3,1.],[0.7,1.],[0.7,-1.]]),
            room(vec![[1.3,-1.],[2.,-1.],[2.,1.],[1.3,1.],[1.3,-1.]]),
        ];
        let groups: Vec<_> = rooms.iter().map(|r| vec![&r.0[0]]).collect();
        crate::walls::SharedWalls::from_cells(&groups)
    };
    let base = scene(0.);
    let edited = scene(0.12);
    let reference = evaluate_shared(&edited, &boundary);
    assert!(reference > 0., "the edit must tilt a shared wall");
    let cache = WallShapeCache::new(&base, &boundary);
    SCORE_WORK.set([0; 2]);
    let actual = cache.evaluate(&edited);
    assert_eq!(SCORE_WORK.get()[0], 1, "only the changed pair is traced again");
    assert!(SCORE_WORK.get()[1] >= 1, "the changed pair's chains are scored again");
    assert_eq!(actual.to_bits(), reference.to_bits());
}

#[test]
fn local_legacy_wall_loss_reuses_pairs_and_skips_cosine_scoring() {
    use crate::{grad_local::LocalGradContext, loss::LossWeights, program::RoomProgram};

    let b = crate::shapes::by_name("shape_e").unwrap().polygon();
    let data: serde_json::Value = serde_json::from_str(include_str!(
        "../../web-tests/fixtures/room-program-six.json"
    )).unwrap();
    let circles: Vec<[f64; 3]> = data["rooms"].as_array().unwrap().iter().map(|r| [
        r["center"][0].as_f64().unwrap(),
        r["center"][1].as_f64().unwrap(),
        r["radius"].as_f64().unwrap(),
    ]).collect();
    let mut program = RoomProgram::new(&b, &circles).unwrap();
    program.set_connections(&[[0, 1], [2, 4]], 0.1).unwrap();
    let (sites, labels) = program.initialize(&b, 80, 4971).unwrap();
    let targets: Vec<_> = program.ratios().iter().map(|r| r * b.unsigned_area()).collect();
    let w = LossWeights {w_wall: 1. / 13., ..Default::default()};
    let ctx = LocalGradContext::new_with_program(&sites, &b, &targets, &labels, &w, Some(&program));

    let mut moved = sites.clone();
    moved[11][0] += 1e-6;
    SCORE_WORK.set([0; 2]);
    crate::walls::OWNER_EDGES_READ.set(0);
    crate::loss::WALL_RINGS_SCORED.set(0);
    let local = ctx.local_total(&moved, 11);
    assert!(local.is_finite() && local > 0.);
    assert_eq!(SCORE_WORK.get(), [0, 0], "L1 alone never traces or scores wall chains for Cos");
    let owner_edges = crate::walls::OWNER_EDGES_READ.get();
    assert!(owner_edges < sites.len(),
        "graph evaluation should visit shared walls and changed cell edges, visited {owner_edges} edges for {} sites",
        sites.len());
    let rings = crate::loss::WALL_RINGS_SCORED.get();
    assert!(rings > 0 && rings < program.ratios().len(),
        "only changed room rings require new legacy wall scores, scored {rings}");
}

fn rectangle() -> Polygon<f64> {
    Polygon::new(LineString::from(vec![(-2.,-1.),(2.,-1.),(2.,1.),(-2.,1.),(-2.,-1.)]),vec![])
}
fn transform(p:[f64;2],angle:f64,scale:f64)->[f64;2]{
    let (s,c)=angle.sin_cos();[scale*(c*p[0]-s*p[1]),scale*(s*p[0]+c*p[1])]
}
fn transformed_boundary(b:&Polygon<f64>,angle:f64,scale:f64)->Polygon<f64>{
    Polygon::new(LineString::from(b.exterior().0.iter().map(|p|{let q=transform([p.x,p.y],angle,scale);(q[0],q[1])}).collect::<Vec<_>>()),vec![])
}
fn split_line(a:[f64;2],b:[f64;2],n:usize)->Vec<[f64;2]>{
    (0..=n).map(|i|{let t=i as f64/n as f64;[a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t]}).collect()
}

#[test]
fn cosine_direction_is_adaptive_and_independent_of_scale_and_subdivision(){
    let boundary=rectangle();
    for degrees in [0_f64,90.,180.]{
        let end=transform([1.,0.],degrees.to_radians(),1.);
        assert!(losses(&[vec![[0.,0.],end]],&boundary).abs()<1e-12,"parallel/perpendicular/reversed walls are all valid");
    }
    let tilted=vec![[0.,0.],transform([1.,0.],30_f64.to_radians(),1.)];
    let value=losses(&[tilted.clone()],&boundary);
    assert!((value-0.1875).abs()<1e-8,"30 degrees gives c²(1−c²) = 0.1875, got {value}");
    assert!((losses(&[tilted.clone(),vec![[-1.,0.],[1.,0.]]],&boundary)-value).abs()<1e-8,"adding a perfect wall cannot dilute the remaining directional error");
    assert!((losses(&[vec![[0.,0.],[0.5,0.5]]],&boundary)-0.25).abs()<1e-12,
        "45 degrees gives c²(1−c²) = 0.25");
    for scale in [0.1,2.,10.]{
        let b=transformed_boundary(&boundary,0.,scale);
        let chain=tilted.iter().map(|&p|transform(p,0.,scale)).collect();
        assert!((losses(&[chain],&b)-value).abs()<1e-8,"uniform size changes must not create perimeter pressure");
    }
    let angle=17_f64.to_radians();
    let b=transformed_boundary(&boundary,angle,1.);
    let chain=tilted.iter().map(|&p|transform(p,angle,1.)).collect();
    assert!((losses(&[chain],&b)-value).abs()<1e-6,"the reference direction rotates with the boundary");
    let diagonal=transformed_boundary(&boundary,45_f64.to_radians(),1.);
    assert!(losses(&[vec![[0.,0.],[0.5,0.5]]],&diagonal)<1e-12,"a diagonal exterior gives a diagonal preferred frame");
    let mixed=Polygon::new(LineString::from(vec![(-2.,-1.),(2.,-1.),(1.,1.),(-2.,1.),(-2.,-1.)]),vec![]);
    let (a,b)=([0.2,-0.6],[1.05,0.6]);
    let whole=losses(&[vec![a,b]],&mixed);
    let divided=split_line(a,b,13);
    assert!((losses(&[divided.clone()],&mixed)-whole).abs()<1e-10,"a physical straight wall is independent of collinear subdivisions even across reference-frame zones");
    assert!((losses(&[divided.into_iter().rev().collect()],&mixed)-whole).abs()<1e-10,"edge winding does not change the score");
    let closed=vec![a,b,[0.5,0.7],[-0.3,-0.4],a];
    let original=losses(&[closed.clone()],&mixed);
    let mut split:Vec<_>=closed.windows(2).flat_map(|p|split_line(p[0],p[1],4).into_iter().take(4)).collect();
    split.rotate_left(2);split.push(split[0]);
    assert!((losses(&[split],&mixed)-original).abs()<1e-10,"a closed wall's arbitrary starting point and subdivisions do not change its direction loss");
    assert_eq!(losses(&[vec![],vec![[0.,0.]],vec![[0.,0.],[0.,0.]]],&boundary),0.,"zero-length geometry is finite and contributes no wall");
}

fn room(points:Vec<[f64;2]>)->MultiPolygon<f64>{
    MultiPolygon(vec![Polygon::new(LineString::from(points.into_iter().map(|p|(p[0],p[1])).collect::<Vec<_>>()),vec![])])
}
