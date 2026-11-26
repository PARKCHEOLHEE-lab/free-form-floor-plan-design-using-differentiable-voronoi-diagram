use super::*;
use geo::{Contains, Point};

#[test]
fn anchors_stay_fixed_during_real_optimizer_steps() {
    let boundary = Polygon::new(LineString::from(vec![(-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.),(-1.,-1.)]), vec![]);
    let mut opt = WasmOpt::build(boundary, 12, vec![0.5,0.5], LossWeights { w_lloyd: 1., ..Default::default() }, 777, 0.01);
    let circles = [[-0.7,0.,0.2],[0.6,0.1,0.4]];
    opt.configure_program(&circles.concat(), &[], 0.1).unwrap();
    let fixed = [[-0.7_f32,0.],[0.6,0.1]];
    assert_eq!(&opt.sites[..2], &fixed, "the first site of each room must be its input center");
    assert_eq!(&opt.room_indices[..2], &[0,1]);
    assert!((opt.target_areas[0]-0.8).abs()<1e-12 && (opt.target_areas[1]-3.2).abs()<1e-12);
    let free_before = opt.sites[2..].to_vec();
    for _ in 0..12 {
        let frame = opt.advance();
        assert_eq!(&opt.sites[..2], &fixed, "fixed centers must not receive gradients, momentum or weight decay");
        for (r,c) in circles.iter().enumerate() {
            assert!(frame.cells.iter().filter(|cell| cell.room==r).any(|cell| {
                Polygon::new(LineString::from(cell.ring.clone()), vec![]).contains(&Point::new(c[0],c[1]))
            }), "room {r} must contain its original user center");
        }
    }
    assert_ne!(&opt.sites[2..], free_before, "free sites still optimize");
    for bad in [vec![], vec![[0.,0.,0.]], vec![[0.,0.,f64::NAN]], vec![[2.,0.,0.1]], vec![[0.9,0.,0.2]], vec![[0.,0.,0.1],[0.,0.,0.2]]] {
        assert!(opt.configure_program(&bad.concat(), &[], 0.1).is_err(), "invalid input must be rejected: {bad:?}");
        assert_eq!(&opt.sites[..2], &fixed, "rejected input must leave the current optimizer intact");
    }
    assert!(opt.configure_program(&[1.,2.], &[], 0.1).is_err());
    assert!(opt.configure_program(&circles.concat(), &[0], 0.1).is_err());
}

#[test]
fn preset_without_a_one_piece_start_keeps_the_kmeans_start() {
    // With 20 sites at seed 145 the notch of shape_a cuts one cell in two, so no start
    // has every room in one piece. A preset has no drawn rooms to adjust.
    let boundary = shapes::by_name("shape_a").unwrap().polygon();
    let mut opt = WasmOpt::build(boundary, 20, vec![0.13,0.30,0.23,0.13,0.21], LossWeights::default(), 145, 0.005);
    let kmeans = opt.room_indices.clone();
    assert_eq!(opt.configure_preset_areas(), Ok(()), "a preset must start without drawn rooms");
    assert_eq!(opt.room_indices, kmeans, "the start stays the KMeans one");
}
