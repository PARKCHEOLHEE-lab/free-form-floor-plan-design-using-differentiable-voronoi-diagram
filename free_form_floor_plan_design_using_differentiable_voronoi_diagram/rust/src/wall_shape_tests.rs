use super::*;
use geo::LineString;

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
