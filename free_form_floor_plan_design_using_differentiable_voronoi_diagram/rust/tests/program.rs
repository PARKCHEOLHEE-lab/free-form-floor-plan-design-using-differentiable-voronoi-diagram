use geo::{LineString, MultiPolygon, Polygon};
use voronoi_floorplan::{
    grad_local::LocalGradContext,
    loss::{floor_plan_loss_with_program, LossWeights},
    program::RoomProgram,
};

fn rect(x: f64, y: f64, xx: f64, yy: f64) -> Polygon<f64> {
    Polygon::new(
        LineString::from(vec![(x, y), (xx, y), (xx, yy), (x, yy), (x, y)]),
        vec![],
    )
}

#[test]
fn initial_room_areas_follow_circle_shares_before_optimization() {
    use geo::{Area, Contains, Point};
    use voronoi_floorplan::voronoi::render_cells;
    let concave=Polygon::new(LineString::from(vec![(-1.,-1.),(1.,-1.),(1.,0.),(0.,0.),(0.,1.),(-1.,1.),(-1.,-1.)]),vec![]);
    let cases=vec![
        (rect(-1.,-1.,1.,1.),vec![[-0.5,0.,0.1],[0.5,0.,0.2]]),
        (rect(-1.,-1.,1.,1.),vec![[-0.5,0.,0.2],[0.5,0.,0.1]]),
        (rect(-1.,-1.,1.,1.),vec![[-0.5,0.,0.15],[0.5,0.,0.15]]),
        (concave,vec![[-0.6,-0.4,0.1],[0.6,-0.5,0.23],[-0.6,0.6,0.14]]),
    ];
    let mut shares=Vec::new();
    for (boundary,circles) in cases {
        let p=RoomProgram::new(&boundary,&circles).unwrap();
        let (sites,labels)=p.initialize(&boundary,80,777).unwrap();
        assert_eq!(p.initialize(&boundary,80,777).unwrap(),(sites.clone(),labels.clone()),"initialization remains deterministic");
        assert_eq!(sites.len(),80);
        assert_eq!(&labels[..circles.len()],(0..circles.len()).collect::<Vec<_>>());
        let cells=render_cells(&sites,&boundary);
        let mut areas=vec![0.;circles.len()];
        for (i,cell) in &cells {areas[labels[*i]]+=cell.unsigned_area();}
        let total=boundary.unsigned_area();
        assert!((areas.iter().sum::<f64>()-total).abs()<1e-7,"all displayed pieces cover the boundary");
        for (r,circle) in circles.iter().enumerate(){
            assert_eq!(sites[r],[circle[0] as f32,circle[1] as f32]);
            assert!(cells.iter().any(|(i,cell)|labels[*i]==r && cell.contains(&Point::new(circle[0],circle[1]))),"each initial room keeps its input center");
            assert!((areas[r]/total-p.ratios()[r]).abs()<=0.05,"room {r}: initial displayed share {} must approach target {}",areas[r]/total,p.ratios()[r]);
        }
        shares.push(areas[0]/total);
    }
    assert!(shares[1]>shares[0]+0.4,"swapping radii swaps actual initial area, not just targets");
    let boundary=rect(-1.,-1.,1.,1.);
    let p=RoomProgram::new(&boundary,&[[-0.5,-0.5,0.1],[0.5,-0.5,0.1],[0.,0.5,0.4]]).unwrap();
    assert_eq!(p.initialize(&boundary,3,777).unwrap().1,vec![0,1,2],"with no free sites, fixed centers take priority over unattainable area targets");
}

#[test]
fn requested_links_change_local_loss_and_gradients_without_legacy_weights() {
    let boundary = rect(-1., -1., 1., 1.);
    let mut p = RoomProgram::new(
        &boundary,
        &[[-0.75, 0., 0.1], [0., 0., 0.1], [0.75, 0., 0.1]],
    )
    .unwrap();
    let sites = vec![
        [-0.75, -0.5],
        [-0.75, 0.5],
        [0., -0.5],
        [0., 0.5],
        [0.75, -0.5],
        [0.75, 0.5],
    ];
    let labels = vec![0, 0, 1, 1, 2, 2];
    let targets = vec![1., 2., 1.];
    let w = LossWeights::default();
    p.set_connections(&[[0, 2]], 0.1).unwrap();
    let global =
        floor_plan_loss_with_program(&sites, &boundary, &targets, &labels, &w, None, Some(&p));
    assert!(
        global.connection > 0.,
        "separated rooms with a requested door must contribute to the loss"
    );
    assert_eq!(
        global.total, global.connection,
        "only the connection term is enabled"
    );
    let ctx =
        LocalGradContext::new_with_program(&sites, &boundary, &targets, &labels, &w, Some(&p));
    assert!((ctx.base_total() - global.total).abs() < 1e-4);
    let missing = ctx.gradients(&sites);
    assert!(missing.iter().flatten().all(|g| g.is_finite()));
    assert!(
        missing.iter().flatten().any(|g| g.abs() > 1e-3),
        "connection loss must change site motion, not only its displayed value"
    );
    let mut moved = sites.clone();
    moved[4][0] += 1e-4;
    let full =
        floor_plan_loss_with_program(&moved, &boundary, &targets, &labels, &w, None, Some(&p))
            .total;
    assert!(
        (ctx.local_total(&moved, 4) - full).abs() < 1e-3,
        "cached perturbed geometry includes the same requested connection"
    );
    p.set_connections(&[[0, 1]], 0.1).unwrap();
    let touching =
        LocalGradContext::new_with_program(&sites, &boundary, &targets, &labels, &w, Some(&p));
    assert_eq!(
        touching.base_breakdown().connection,
        0.,
        "a sufficient shared wall satisfies the link"
    );
    assert_ne!(
        touching.gradients(&sites),
        missing,
        "changing only the linked pair changes the gradient"
    );
    p.set_connections(&[[0, 1]], 1.5).unwrap();
    assert_eq!(
        floor_plan_loss_with_program(&sites, &boundary, &targets, &labels, &w, None, Some(&p))
            .connection,
        0.,
        "collinear pieces form one wall long enough for the door"
    );
    p.set_connections(&[[0, 1]], 0.1).unwrap();
    let cells = vec![
        MultiPolygon(vec![rect(0., 0., 1., 1.)]),
        MultiPolygon(vec![rect(1., 1., 2., 2.)]),
        MultiPolygon(vec![rect(3., 0., 4., 1.)]),
    ];
    assert!(
        p.connection_loss(&cells) > 0.,
        "point contact cannot hold a door"
    );
    let adjacent = vec![
        MultiPolygon(vec![rect(0., 0., 1., 1.)]),
        MultiPolygon(vec![rect(1., 0., 2., 1.)]),
        cells[2].clone(),
    ];
    assert_eq!(p.connection_loss(&adjacent), 0.);
    p.set_connections(&[[0, 1]], 1.2).unwrap();
    assert!(
        p.connection_loss(&adjacent) > 0.,
        "the shared wall must fit the door width"
    );
    for (links, width) in [
        (vec![[0, 3]], 0.1),
        (vec![[1, 1]], 0.1),
        (vec![[0, 1], [1, 0]], 0.1),
        (vec![[0, 1]], 0.),
        (vec![[0, 1]], f64::NAN),
    ] {
        assert!(
            p.set_connections(&links, width).is_err(),
            "invalid connections must be rejected"
        );
    }
    p.set_connections(&[], 0.1).unwrap();
    assert_eq!(
        p.connection_loss(&[]),
        0.,
        "no links need no connection geometry"
    );
}
