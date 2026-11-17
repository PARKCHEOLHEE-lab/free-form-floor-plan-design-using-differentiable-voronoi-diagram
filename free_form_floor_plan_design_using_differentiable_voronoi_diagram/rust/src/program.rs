//! User room intentions in the optimizer's boundary coordinates. Center sites are
//! the first parameters and must be excluded from AdamW, including its decay.
use geo::{Area, ConvexHull, Contains, Line, MultiPoint, MultiPolygon, Point, Polygon, Simplify};

pub struct RoomProgram {
    centers: Vec<[f32; 2]>,
    ratios: Vec<f64>,
    links: Vec<[usize; 2]>,
    door_width: f64,
}

impl RoomProgram {
    pub fn set_connections(&mut self, links: &[[usize; 2]], width: f64) -> Result<(), String> {
        if !width.is_finite() || width <= 0. {
            return Err("Door width must be positive and finite.".into());
        }
        let mut checked = Vec::new();
        for &[a, b] in links {
            if a == b || a >= self.centers.len() || b >= self.centers.len() {
                return Err("A connection needs two different existing rooms.".into());
            }
            let pair = if a < b { [a, b] } else { [b, a] };
            if checked.contains(&pair) {
                return Err("Duplicate room connection.".into());
            }
            checked.push(pair);
        }
        self.links = checked;
        self.door_width = width;
        Ok(())
    }

    pub fn has_connections(&self) -> bool {
        !self.links.is_empty()
    }

    /// A missing connection attracts the rooms across their gap. Once they
    /// touch, the remaining penalty grows a straight shared wall to door width.
    #[allow(deprecated)]
    pub fn connection_loss(&self, rooms: &[MultiPolygon<f64>]) -> f32 {
        use geo::{
            line_intersection::{line_intersection, LineIntersection},
            EuclideanDistance,
        };
        if self.links.is_empty() {
            return 0.;
        }
        // A zero epsilon disables geo's simplifier. Its smallest positive value
        // joins collinear pieces without straightening actual bends in the wall.
        let edges: Vec<Vec<Line<f64>>> = rooms
            .iter()
            .map(|room| {
                room.iter()
                    .flat_map(|p| std::iter::once(p.exterior()).chain(p.interiors()))
                    .flat_map(|ring| {
                        ring.simplify(&f64::MIN_POSITIVE)
                            .lines()
                            .collect::<Vec<_>>()
                    })
                    .collect()
            })
            .collect();
        self.links
            .iter()
            .map(|&[a, b]| {
                let gap = rooms[a].euclidean_distance(&rooms[b]);
                let mut shared: f64 = 0.;
                if gap == 0. {
                    for &x in &edges[a] {
                        for &y in &edges[b] {
                            if let Some(LineIntersection::Collinear { intersection }) =
                                line_intersection(x, y)
                            {
                                shared = shared.max(
                                    (intersection.end.x - intersection.start.x)
                                        .hypot(intersection.end.y - intersection.start.y),
                                );
                            }
                        }
                    }
                }
                let deficit = (self.door_width - shared).max(0.);
                ((gap + deficit) / self.door_width).powi(2)
            })
            .sum::<f64>() as f32
    }

    #[allow(deprecated)] // geo 0.30's point-to-ring distance uses this trait.
    pub fn new(boundary: &Polygon<f64>, circles: &[[f64; 3]]) -> Result<Self, String> {
        use geo::EuclideanDistance;
        if circles.is_empty() {
            return Err("Draw at least one room.".into());
        }
        let mut centers = Vec::new();
        let mut areas = Vec::new();
        for &[x, y, r] in circles {
            if ![x, y, r].iter().all(|v| v.is_finite()) || r <= 0. || r * r == 0. {
                return Err("Room circles need finite coordinates and a positive radius.".into());
            }
            let point = Point::new(x, y);
            if !boundary.contains(&point)
                || std::iter::once(boundary.exterior())
                    .chain(boundary.interiors())
                    .any(|ring| point.euclidean_distance(ring) < r)
            {
                return Err("Keep the whole circle inside the boundary.".into());
            }
            let center = [x as f32, y as f32];
            if !boundary.contains(&Point::new(center[0] as f64,center[1] as f64)) {
                return Err("Room center is too close to the boundary at engine precision.".into());
            }
            if centers.contains(&center) {
                return Err("Room centers must be distinct at engine precision.".into());
            }
            centers.push(center);
            areas.push(r * r);
        }
        let sum: f64 = areas.iter().sum();
        Ok(Self {
            centers,
            ratios: areas.into_iter().map(|a| a / sum).collect(),
            links: Vec::new(),
            door_width: 0.,
        })
    }

    pub fn fixed_count(&self) -> usize {
        self.centers.len()
    }
    pub fn ratios(&self) -> &[f64] {
        &self.ratios
    }

    /// Preserve the existing seeded site cloud, excluding duplicates of fixed
    /// centers. Each remaining site starts in the nearest user-specified room.
    pub fn initialize(
        &self,
        boundary: &Polygon<f64>,
        count: usize,
        seed: u64,
    ) -> Result<(Vec<[f32; 2]>, Vec<usize>), String> {
        if count < self.centers.len() {
            return Err("Use at least one site per room.".into());
        }
        let mut sites = self.centers.clone();
        sites.extend(
            crate::init::initialize_sites(boundary, count, seed)
                .into_iter()
                .filter(|p| !self.centers.contains(p))
                .take(count - self.centers.len()),
        );
        let hull = MultiPoint::from(sites.iter().map(|p|(p[0] as f64,p[1] as f64)).collect::<Vec<_>>()).convex_hull();
        if hull.unsigned_area()==0. { return Err("Use at least three non-collinear sites. Increase the site budget.".into()); }
        let labels = sites
            .iter()
            .map(|p| {
                self.centers
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let distance = |q: &[f32; 2]| (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2);
                        distance(a).total_cmp(&distance(b))
                    })
                    .unwrap()
                    .0
            })
            .collect();
        Ok((sites, labels))
    }
}
