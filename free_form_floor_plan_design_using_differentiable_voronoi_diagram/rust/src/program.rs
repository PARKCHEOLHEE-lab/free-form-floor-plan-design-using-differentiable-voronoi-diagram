//! User room intentions in the optimizer's boundary coordinates. Center sites are
//! the first parameters and must be excluded from AdamW, including its decay.
use geo::{Contains, Point, Polygon};

pub struct RoomProgram {
    centers: Vec<[f32; 2]>,
    ratios: Vec<f64>,
}

impl RoomProgram {
    #[allow(deprecated)] // geo 0.30's point-to-ring distance uses this trait.
    pub fn new(boundary: &Polygon<f64>, circles: &[[f64; 3]]) -> Result<Self, String> {
        use geo::EuclideanDistance;
        if circles.is_empty() { return Err("Draw at least one room.".into()); }
        let mut centers = Vec::new();
        let mut areas = Vec::new();
        for &[x,y,r] in circles {
            if ![x,y,r].iter().all(|v| v.is_finite()) || r <= 0. || r*r == 0. {
                return Err("Room circles need finite coordinates and a positive radius.".into());
            }
            let point = Point::new(x,y);
            if !boundary.contains(&point) || std::iter::once(boundary.exterior()).chain(boundary.interiors())
                .any(|ring| point.euclidean_distance(ring) < r) {
                return Err("Keep the whole circle inside the boundary.".into());
            }
            let center = [x as f32, y as f32];
            if centers.contains(&center) { return Err("Room centers must be distinct at engine precision.".into()); }
            centers.push(center);
            areas.push(r*r);
        }
        let sum: f64 = areas.iter().sum();
        Ok(Self { centers, ratios: areas.into_iter().map(|a| a/sum).collect() })
    }

    pub fn fixed_count(&self) -> usize { self.centers.len() }
    pub fn ratios(&self) -> &[f64] { &self.ratios }

    /// Preserve the existing seeded site cloud, excluding duplicates of fixed
    /// centers. Each remaining site starts in the nearest user-specified room.
    pub fn initialize(&self, boundary: &Polygon<f64>, count: usize, seed: u64) -> Result<(Vec<[f32; 2]>, Vec<usize>), String> {
        if count < self.centers.len() { return Err("Use at least one site per room.".into()); }
        let mut sites = self.centers.clone();
        sites.extend(crate::init::initialize_sites(boundary, count, seed).into_iter()
            .filter(|p| !self.centers.contains(p)).take(count-self.centers.len()));
        let labels = sites.iter().map(|p| self.centers.iter().enumerate().min_by(|(_,a),(_,b)| {
            let distance = |q: &[f32; 2]| (p[0]-q[0]).powi(2)+(p[1]-q[1]).powi(2);
            distance(a).total_cmp(&distance(b))
        }).unwrap().0).collect();
        Ok((sites,labels))
    }
}
