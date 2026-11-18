use super::*;
use geo::{LineString,MultiPolygon};

#[test]
fn mixed_wall_edit_reuses_unchanged_pairs_without_scoring_unused_direction() {
    let boundary = rectangle();
    let scene = |offset| {
        let rooms = [
            room(vec![[-2.,-1.],[-1.,-1.],[-1.,1.],[-2.,1.],[-2.,-1.]]),
            room(vec![[-1.,-1.],[0.,-1.],[0.,0.],[offset,0.],[offset,1.],[-1.,1.],[-1.,-1.]]),
            room(vec![[0.,-1.],[0.7,-1.],[0.7,1.],[offset,1.],[offset,0.],[0.,0.],[0.,-1.]]),
            room(vec![[0.7,-1.],[1.3,-1.],[1.3,1.],[0.7,1.],[0.7,-1.]]),
            room(vec![[1.3,-1.],[2.,-1.],[2.,1.],[1.3,1.],[1.3,-1.]]),
        ];
        let groups: Vec<_> = rooms.iter().map(|r| vec![&r.0[0]]).collect();
        crate::walls::SharedWalls::from_cells(&groups)
    };
    let base = scene(0.);
    let edited = scene(0.12);
    let reference = evaluate_shared(&edited, &boundary).1;
    assert!(reference > 0., "the edit must introduce a real step");
    let cache = WallShapeCache::new(&base, &boundary, false, true);
    SCORE_WORK.set([0; 2]);
    let actual = cache.evaluate(&edited);
    assert_eq!(SCORE_WORK.get(), [1, 0], "only the changed pair is traced, and no direction is scored");
    assert_eq!(actual.0, 0.);
    assert_eq!(actual.1.to_bits(), reference.to_bits());
}

#[test]
fn local_legacy_wall_loss_reuses_pairs_and_skips_cosine_scoring() {
    use crate::{grad_local::LocalGradContext, loss::LossWeights, program::RoomProgram, voronoi};

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
    let w = LossWeights {w_wall: 1. / 13., w_jog: 0.01, ..Default::default()};
    let ctx = LocalGradContext::new_with_program(&sites, &b, &targets, &labels, &w, Some(&program));

    let mut moved = sites.clone();
    moved[11][0] += 1e-6;
    let cells = voronoi::compute_cells(&moved, &b, None).cells_sorted;
    let all = crate::walls::SharedWallCells::new(&cells, &labels).snapshot();
    SCORE_WORK.set([0; 2]);
    crate::walls::OWNER_EDGES_READ.set(0);
    crate::loss::WALL_RINGS_SCORED.set(0);
    let local = ctx.local_total(&moved, 11);
    let [pairs_scored, directions_scored] = SCORE_WORK.get();
    assert!(local.is_finite() && local > 0.);
    assert!(pairs_scored > 0 && pairs_scored < all.indexed_pairs().count(),
        "one finite difference must score a strict subset of the wall pairs, scored {pairs_scored}");
    assert_eq!(directions_scored, 0, "legacy wall loss never uses the cosine score");
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
        assert!(losses(&[vec![[0.,0.],end]],&boundary).0.abs()<1e-12,"parallel/perpendicular/reversed walls are all valid");
    }
    let tilted=vec![[0.,0.],transform([1.,0.],30_f64.to_radians(),1.)];
    let value=losses(&[tilted.clone()],&boundary).0;
    assert!((value-0.75).abs()<1e-8,"30 degrees has normalized cos-squared penalty 0.75, got {value}");
    assert!((losses(&[tilted.clone(),vec![[-1.,0.],[1.,0.]]],&boundary).0-value).abs()<1e-8,"adding a perfect wall cannot dilute the remaining directional error");
    assert!((losses(&[vec![[0.,0.],[0.5,0.5]]],&boundary).0-1.).abs()<1e-12,"45 degrees is the maximum directional error");
    for scale in [0.1,2.,10.]{
        let b=transformed_boundary(&boundary,0.,scale);
        let chain=tilted.iter().map(|&p|transform(p,0.,scale)).collect();
        assert!((losses(&[chain],&b).0-value).abs()<1e-8,"uniform size changes must not create perimeter pressure");
    }
    let angle=17_f64.to_radians();
    let b=transformed_boundary(&boundary,angle,1.);
    let chain=tilted.iter().map(|&p|transform(p,angle,1.)).collect();
    assert!((losses(&[chain],&b).0-value).abs()<1e-6,"the reference direction rotates with the boundary");
    let diagonal=transformed_boundary(&boundary,45_f64.to_radians(),1.);
    assert!(losses(&[vec![[0.,0.],[0.5,0.5]]],&diagonal).0<1e-12,"a diagonal exterior gives a diagonal preferred frame");
    let mixed=Polygon::new(LineString::from(vec![(-2.,-1.),(2.,-1.),(1.,1.),(-2.,1.),(-2.,-1.)]),vec![]);
    let (a,b)=([0.2,-0.6],[1.05,0.6]);
    let whole=losses(&[vec![a,b]],&mixed).0;
    let divided=split_line(a,b,13);
    assert!((losses(&[divided.clone()],&mixed).0-whole).abs()<1e-10,"a physical straight wall is independent of collinear subdivisions even across reference-frame zones");
    assert!((losses(&[divided.into_iter().rev().collect()],&mixed).0-whole).abs()<1e-10,"edge winding does not change the score");
    let closed=vec![a,b,[0.5,0.7],[-0.3,-0.4],a];
    let original=losses(&[closed.clone()],&mixed).0;
    let mut split:Vec<_>=closed.windows(2).flat_map(|p|split_line(p[0],p[1],4).into_iter().take(4)).collect();
    split.rotate_left(2);split.push(split[0]);
    assert!((losses(&[split],&mixed).0-original).abs()<1e-10,"a closed wall's arbitrary starting point and subdivisions do not change its direction loss");
    assert_eq!(losses(&[vec![],vec![[0.,0.]],vec![[0.,0.],[0.,0.]]],&boundary).0,0.,"zero-length geometry is finite and contributes no wall");
}

fn step(h:f64)->Vec<[f64;2]>{vec![[-0.8,-0.5],[0.,-0.5],[0.,-0.5+h],[0.8,-0.5+h]]}
fn room(points:Vec<[f64;2]>)->MultiPolygon<f64>{
    MultiPolygon(vec![Polygon::new(LineString::from(points.into_iter().map(|p|(p[0],p[1])).collect::<Vec<_>>()),vec![])])
}

#[test]
fn jog_loss_aligns_parallel_runs_of_the_same_shared_wall(){
    let boundary=rectangle();
    let small=losses(&[step(0.02)],&boundary);
    let large=losses(&[step(0.04)],&boundary);
    assert_eq!(large.0,0.,"an orthogonal step is already direction-aligned");
    assert!(small.1>0. && large.1>small.1,"jog must measure parallel-line offset, not just edge angle");
    let h=0.04;
    let derivative=(losses(&[step(h+1e-6)],&boundary).1-losses(&[step(h-1e-6)],&boundary).1)/2e-6;
    assert!(derivative>0.,"gradient descent must shrink the step, not enlarge its short edge");
    assert_eq!(losses(&[step(0.)],&boundary).1,0.);
    for points in [vec![[-1.,0.],[0.,0.],[0.,1.]],vec![[-1.,0.],[0.,0.],[0.,1.],[-1.,1.]]]{
        assert_eq!(losses(&[points],&boundary).1,0.,"major L and U corners are retained");
    }
    for degrees in [7_f64,17.,33.]{
        let a=degrees.to_radians();let b=transformed_boundary(&boundary,a,0.3);
        let points=vec![[-0.9,-0.1],[0.1,-0.1],[0.1,0.9]].into_iter().map(|p|transform(p,a,0.3)).collect();
        assert_eq!(losses(&[points],&b).1,0.,"one line per heading has exactly zero variance, including rotated/scaled L corners");
    }
    assert_eq!(losses(&[vec![[0.,-0.5],[0.,0.5]],vec![[0.2,-0.5],[0.2,0.5]]],&boundary).1,0.,"separate physical walls cannot attract each other");
    let sheared:Vec<_>=step(h).into_iter().map(|p|[p[0],p[1]+0.5*p[0]]).collect();
    assert!((losses(&[sheared],&boundary).1-large.1).abs()<1e-10,"compare supporting-line intercepts, not the locations of segment midpoints");
    let angle=17_f64.to_radians();
    let b=transformed_boundary(&boundary,angle,3.);
    let chain=step(h).into_iter().map(|p|transform(p,angle,3.)).collect();
    assert!((losses(&[chain],&b).1-large.1).abs()<1e-7,"jog is independent of uniform scale and rotation");
    let bump=vec![[0.,-1.],[0.,-0.3],[h,-0.3],[h,0.3],[0.,0.3],[0.,1.]];
    let mut left=vec![[-2.,-1.]];left.extend(bump.clone());left.extend([[-2.,1.],[-2.,-1.]]);
    let mut right:Vec<_>=bump.iter().copied().rev().collect();right.extend([[2.,-1.],[2.,1.],[0.,1.]]);
    let rooms=vec![room(left),room(right)];
    let actual=evaluate(&rooms,&boundary);
    assert!(actual.1>0.,"extract the connected shared wall including its transverse connectors");
    assert!((actual.1*2.-losses(&[bump],&boundary).1).abs()<1e-10,"physical shared walls count once with fixed room-count normalization");
    let reversed:Vec<_>=rooms.iter().map(|r|MultiPolygon(r.0.iter().map(|p|Polygon::new(LineString::from(p.exterior().0.iter().rev().copied().collect::<Vec<_>>()),vec![])).collect())).collect();
    assert!((evaluate(&reversed,&boundary).1-actual.1).abs()<1e-10);
    assert_eq!(evaluate(&rooms[..1],&boundary),(0.,0.),"external walls are fixed geometry, not shared-wall fitting targets");
    let stripes=vec![room(vec![[-2.,-1.],[-0.5,-1.],[-0.5,1.],[-2.,1.],[-2.,-1.]]),room(vec![[-0.5,-1.],[0.5,-1.],[0.5,1.],[-0.5,1.],[-0.5,-1.]]),room(vec![[0.5,-1.],[2.,-1.],[2.,1.],[0.5,1.],[0.5,-1.]])];
    assert_eq!(evaluate(&stripes,&boundary),(0.,0.),"parallel walls between different room pairs remain independent");
}

#[test]
fn vanishing_transverse_runs_have_continuous_jog_loss(){
    let boundary=rectangle();
    let value=|eps:f64|losses(&[vec![[0.,-1.],[eps,-1.],[eps,0.],[2.*eps,0.],[2.*eps,1.]]],&boundary).1;
    assert_eq!(value(0.),0.);
    let coarse=value(1e-4);let fine=value(1e-6);
    assert!(fine<1e-4 && fine<coarse*0.02,"a disappearing offset cannot retain a finite penalty: coarse={coarse}, fine={fine}");
    let derivative=(value(1.01e-6)-value(0.99e-6))/2e-8;
    assert!(derivative>0. && derivative<100.,"tiny runs must produce a bounded descent gradient, got {derivative}");
}

#[test]
fn visible_steps_cannot_get_cheaper_by_run_imbalance_or_large_offset(){
    let boundary=rectangle();
    let original=losses(&[step(0.04)],&boundary).1;
    let enlarged=vec![[-0.8,-0.5],[0.,-0.5],[0.,-0.44],[0.1,-0.44]];
    let enlarged_loss=losses(&[enlarged],&boundary).1;
    assert!(enlarged_loss>original,"growing a visible normalized offset0.04->0.06 cannot get cheaper by shortening one run to0.1: original={original}, enlarged={enlarged_loss}");
    let derivative=|h:f64|(losses(&[step(h+1e-6)],&boundary).1-losses(&[step(h-1e-6)],&boundary).1)/2e-6;
    let small = derivative(0.04);
    // Both offsets fit inside the test boundary, including the far end
    // where a bounded response can lose nearly all of its gradient.
    for height in [0.4, 1.2] {
        let large = derivative(height);
        assert!(large >= small * 0.5,
            "step {height} still needs a shrinking gradient: small={small}, large={large}");
    }
}


// Independent geometric measurement, not the wall/jog objective. Count the
// offsets of three-edge stairs with visible arms in one room-pair chain.
fn geometric_stairs(cells:&[(usize,Polygon<f64>)],count:usize)->(f64,f64){
    let groups:Vec<_>=(0..count).map(|r|MultiPolygon(cells.iter().filter(|(room,_)|*room==r).map(|(_,p)|p.clone()).collect())).collect();
    let edges:Vec<_>=groups.iter().map(crate::walls::boundary_segments).collect();
    let mut heights=Vec::new();
    for r in 0..count{for t in r+1..count{
        let shared=crate::walls::shared_wall_segments(&edges[r],&edges[t]);
        for chain in wall_chains(&shared,1e-8){
            let p=canonical_chain(&chain,1e-8);
            let mut v:Vec<_>=p.windows(2).map(|p|[p[1][0]-p[0][0],p[1][1]-p[0][1]]).collect();
            if p.len()>3&&p.first()==p.last(){let tail:Vec<_>=v.iter().copied().take(2).collect();v.extend(tail);}
            for e in v.windows(3){
                let l=[e[0][0].hypot(e[0][1]),e[1][0].hypot(e[1][1]),e[2][0].hypot(e[2][1])];
                if l[0]<0.04||l[2]<0.04||l[1]<1e-6{continue;}
                let dot=|a:usize,b:usize|(e[a][0]*e[b][0]+e[a][1]*e[b][1])/(l[a]*l[b]);let limit=10_f64.to_radians();
                if dot(0,2)<limit.cos()||dot(0,1).abs()>limit.sin()||dot(1,2).abs()>limit.sin(){continue;}
                heights.push((e[0][0]*e[1][1]-e[0][1]*e[1][0]).abs()/l[0]);
            }
        }
    }}
    (heights.iter().copied().fold(0.,f64::max),heights.iter().sum())
}
fn frozen_bad_cells()->Vec<(usize,Polygon<f64>)>{
    let data:serde_json::Value=serde_json::from_str(include_str!("../../web-tests/fixtures/room-program-six-bad.json")).unwrap();
    data["cells"].as_array().unwrap().iter().map(|c|{
        let ring=c["ring"].as_array().unwrap().iter().map(|p|(p[0].as_f64().unwrap(),p[1].as_f64().unwrap())).collect::<Vec<_>>();
        (c["room"].as_u64().unwrap() as usize,Polygon::new(LineString::from(ring),vec![]))
    }).collect()
}
#[test]
fn independent_stair_measurement_detects_the_frozen_reported_layout(){
    let measured=geometric_stairs(&frozen_bad_cells(),6);
    assert!((measured.0-0.18951299813428618).abs()<1e-8,"independent historical maximum offset, got {measured:?}");
    assert!(// Reversing nearly parallel arms changes the projected total by less
    // than0.02 screen pixels; keep the historical geometry/value unchanged.
    (measured.1-0.6976662135381002).abs()<1e-4,"independent historical total offset, got {measured:?}");
}
#[test]
fn six_room_raw_stairs_improve_with_orientation_area_centers_and_connectivity(){
    use geo::{Contains,Point};
    use crate::{grad_local::LocalGradContext,loss::{LossWeights,WallLossMode},optim::AdamW,program::RoomProgram,voronoi};
    let data:serde_json::Value=serde_json::from_str(include_str!("../../web-tests/fixtures/room-program-six.json")).unwrap();
    let circles:Vec<[f64;3]>=data["rooms"].as_array().unwrap().iter().map(|r|[r["center"][0].as_f64().unwrap(),r["center"][1].as_f64().unwrap(),r["radius"].as_f64().unwrap()]).collect();
    let b=crate::shapes::by_name("shape_e").unwrap().polygon();let mut p=RoomProgram::new(&b,&circles).unwrap();p.set_connections(&[],0.1).unwrap();
    let (mut sites,labels)=p.initialize(&b,80,4971).unwrap();let fixed=sites[..6].to_vec();let targets:Vec<_>=p.ratios().iter().map(|r|r*b.unsigned_area()).collect();
    let w=LossWeights{wall_mode:WallLossMode::CosineSquared,w_wall:1.,w_jog:0.35,w_area:0.15/0.49,w_lloyd:0.01/5.3,w_topo:0.2/4.,..Default::default()};
    let mut opt=AdamW::new(74,0.005);
    for iteration in 1..=500{if iteration==300{opt.set_lr(0.004);}let ctx=LocalGradContext::new_with_program(&sites,&b,&targets,&labels,&w,Some(&p));let g=ctx.gradients(&sites);assert!(g.iter().flatten().all(|v|v.is_finite()));opt.step(&mut sites[6..],&g[6..]);}
    let rendered=voronoi::render_cells(&sites,&b);let cells:Vec<_>=rendered.into_iter().map(|(i,c)|(labels[i],c)).collect();
    let reference=geometric_stairs(&frozen_bad_cells(),6);let measured=geometric_stairs(&cells,6);
    let bd=LocalGradContext::new_with_program(&sites,&b,&targets,&labels,&w,Some(&p)).base_breakdown();
    eprintln!("six-room Raw: max/total={measured:?}, bad={reference:?}, angle={},area={}",bd.wall,bd.area);
    assert!(measured.0<=reference.0*0.7&&measured.1<=reference.1*0.7,"actual geometric maximum and total stairs must shrink by at least30%");
    assert!(bd.wall<=0.02,"stairs cannot disappear merely by bending walls away from adaptive axes, got {}",bd.wall);
    assert_eq!(&sites[..6],fixed);
    let mut sum=0.;
    for r in 0..6{
        let polygons:Vec<_>=cells.iter().filter(|(room,_)|*room==r).map(|(_,p)|p).collect();
        let area:f64=polygons.iter().map(|p|p.unsigned_area()).sum();sum+=area;
        assert!((area/b.unsigned_area()-p.ratios()[r]).abs()<=0.05,"room{r} actual area share must stay within5pp");
        let union=geo::algorithm::unary_union(polygons.iter().copied());assert_eq!(union.0.len(),1,"room{r} stays one region");
        assert!(union.contains(&Point::new(circles[r][0],circles[r][1])),"room{r} contains its fixed center");
    }
    assert!((sum-b.unsigned_area()).abs()<1e-7,"all actual pieces still cover the boundary");
}

#[test]
fn a_nearly_straight_subdivision_cannot_change_an_existing_steps_cost(){
    let boundary=rectangle();
    let whole=vec![[-0.8,-0.5],[0.,-0.5],[0.,-0.4],[0.8,-0.4]];
    let base=losses(&[whole],&boundary).1;
    let split=|eps:f64|losses(&[vec![[-0.8,-0.5],[-0.4,-0.5+eps],[0.,-0.5],[0.,-0.4],[0.8,-0.4]]],&boundary).1;
    let near=split(1e-8);
    assert!((near-base).abs()<1e-6,"tiny bend cannot multiply an existing step cost: base={base}, near={near}");
    assert!((split(1e-6)-base).abs()<1e-4);
}

#[test]
fn jog_cost_applies_when_the_neighbor_changes_at_a_junction() {
    let b = rectangle();
    let score = |h| {
        let lower = room(vec![
            [-2., -1.],
            [2., -1.],
            [2., h],
            [0., h],
            [0., 0.],
            [-2., 0.],
            [-2., -1.],
        ]);
        let upper_left = room(vec![
            [-2., 0.],
            [0., 0.],
            [0., h],
            [0., 1.],
            [-2., 1.],
            [-2., 0.],
        ]);
        let upper_right = room(vec![[0., h], [2., h], [2., 1.], [0., 1.], [0., h]]);
        let mut groups = vec![
            vec![&lower.0[0]],
            vec![&upper_left.0[0]],
            vec![&upper_right.0[0]],
        ];
        let walls = crate::walls::SharedWalls::from_cells(&groups);
        groups[2].clear();
        let missing = crate::walls::SharedWalls::from_cells(&groups);
        assert!(walls.indexed_pairs().count() > missing.indexed_pairs().count());
        let without = evaluate_shared(&missing, &b).1;
        let with = evaluate_shared(&walls, &b).1;
        let base = WallShapeCache::new(&walls, &b, false, true);
        assert_eq!(base.evaluate(&missing).1.to_bits(), without.to_bits(),
            "a disappearing contact invalidates the old junction score");
        let base = WallShapeCache::new(&missing, &b, false, true);
        assert_eq!(base.evaluate(&walls).1.to_bits(), with.to_bits(),
            "a new contact invalidates the old junction score");
        if h > 0. {
            assert!(with > without, "the contact change must remove a real step");
        }
        with
    };
    let visible = score(0.12);
    assert!(
        visible > 0.001,
        "a visible step belongs to the same room even with different neighbors: {visible}"
    );
    assert!(
        score(0.06) < visible,
        "decreasing the step must lower the loss"
    );
    assert_eq!(
        score(0.),
        0.,
        "a straight wall through a junction costs nothing"
    );
}

#[test]
fn a_junction_step_cannot_pivot_its_arm_to_escape_the_fit() {
    let boundary = rectangle();
    let score = |toe| {
        // One room's lower wall runs across two neighbors. Only the far
        // endpoint of the second arm moves; the connector remains the same.
        let h = 0.12;
        let end = h + toe;
        let lower = room(vec![
            [-2., -1.],
            [2., -1.],
            [2., end],
            [0., h],
            [0., 0.],
            [-2., 0.],
            [-2., -1.],
        ]);
        let upper_left = room(vec![
            [-2., 0.],
            [0., 0.],
            [0., h],
            [0., 1.],
            [-2., 1.],
            [-2., 0.],
        ]);
        let upper_right = room(vec![[0., h], [2., end], [2., 1.], [0., 1.], [0., h]]);
        let groups = vec![
            vec![&lower.0[0]],
            vec![&upper_left.0[0]],
            vec![&upper_right.0[0]],
        ];
        let walls = crate::walls::SharedWalls::from_cells(&groups);
        evaluate_shared(&walls, &boundary).1
    };
    let flat = score(0.);
    let pivoted = score(0.22);
    assert!(pivoted>flat*1.25,"rotating the arm away from its preferred line must increase the junction fit cost: flat={flat}, pivoted={pivoted}");
}

#[test]
fn genuine_u_corner_through_a_neighbor_change_has_no_step_cost() {
    let boundary = rectangle();
    let around = room(vec![
        [-2., -1.],
        [-0.4, -1.],
        [-0.4, 0.],
        [0., 0.],
        [0.4, 0.],
        [0.4, -1.],
        [2., -1.],
        [2., 1.],
        [-2., 1.],
        [-2., -1.],
    ]);
    let left = room(vec![
        [-0.4, -1.],
        [0., -1.],
        [0., 0.],
        [-0.4, 0.],
        [-0.4, -1.],
    ]);
    let right = room(vec![[0., -1.], [0.4, -1.], [0.4, 0.], [0., 0.], [0., -1.]]);
    let groups = vec![vec![&around.0[0]], vec![&left.0[0]], vec![&right.0[0]]];
    let shared = crate::walls::SharedWalls::from_cells(&groups);

    assert_eq!(
        evaluate_shared(&shared, &boundary).1,
        0.,
        "a necessary U corner remains a U even where the opposite room changes"
    );
}
