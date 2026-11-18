use super::*;
use geo::{LineString,MultiPolygon};

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
    let (small,large)=(derivative(0.04),derivative(0.4));
    assert!(large>=small*0.5,"large steps still need a useful shrinking gradient: small={small}, large={large}");
}
