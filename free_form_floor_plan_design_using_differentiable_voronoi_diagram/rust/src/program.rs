//! User room intentions in the optimizer's boundary coordinates. Center sites are
//! the first parameters and must be excluded from AdamW, including its decay.
use geo::{Area, ConvexHull, Contains, MultiPoint, MultiPolygon, Point, Polygon};

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
        use geo::EuclideanDistance;
        if self.links.is_empty() { return 0.; }
        let edges: Vec<_> = rooms.iter().map(crate::walls::boundary_segments).collect();
        self.links
            .iter()
            .map(|&[a, b]| {
                let gap = rooms[a].euclidean_distance(&rooms[b]);
                let shared = if gap == 0. {
                    crate::walls::shared_wall_segments(&edges[a], &edges[b]).iter()
                        .map(|line| (line.end.x-line.start.x).hypot(line.end.y-line.start.y))
                        .fold(0., f64::max)
                } else { 0. };
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
    /// centers. Assign the remaining cells near their room centers while fitting
    /// the requested shares, measured from every displayed clipped piece.
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
        let labels = self.initial_labels(&sites, boundary)?;
        Ok((sites, labels))
    }

    fn initial_labels(&self, sites: &[[f32;2]], boundary: &Polygon<f64>) -> Result<Vec<usize>,String> {
        let (k,n)=(self.centers.len(),sites.len());
        let mut pieces=vec![Vec::new();n];
        for (i,cell) in crate::voronoi::render_cells(sites,boundary){pieces[i].push(cell);}
        let cell_areas:Vec<f64>=pieces.iter().map(|parts|parts.iter().map(Area::unsigned_area).sum()).collect();
        let connected=|labels:&[usize]|(0..k).all(|r|{
            let cells:Vec<_>=pieces.iter().enumerate().filter(|(i,_)|labels[*i]==r).flat_map(|(_,p)|p.iter()).collect();
            geo::algorithm::unary_union(cells).0.len()==1
        });
        let failure=||"Could not initialize connected rooms with these cells. Increase sites or adjust circles.".to_string();
        if k==1 {return Ok(vec![0;n]);}
        if n==k {
            let labels:Vec<_>=(0..k).collect();
            return if connected(&labels){Ok(labels)}else{Err(failure())};
        }
        // Raw Voronoi neighbors may meet outside a concave boundary. Only an
        // actual union across a shared clipped edge supplies a growth path.
        let points:Vec<_>=sites.iter().map(|p|[p[0] as f64,p[1] as f64]).collect();
        let neighbors=crate::voronoi::site_neighbors(&points,boundary);
        let mut graph=vec![Vec::new();n];
        let mut span:f64=0.;
        for i in 0..n {for &j in &neighbors[i] {if j>i {
            let cells:Vec<_>=pieces[i].iter().chain(&pieces[j]).collect();
            if geo::algorithm::unary_union(cells).0.len()<pieces[i].len()+pieces[j].len(){
                let distance=(points[i][0]-points[j][0]).hypot(points[i][1]-points[j][1]);
                graph[i].push((j,distance));graph[j].push((i,distance));span=span.max(distance*n as f64);
            }
        }}}
        // Multi-source shortest-path expansion: a cell can only join a room
        // through a cell that already belongs to that room. Seeds cannot move.
        let assign=|weights:&[f64]|->Option<Vec<usize>>{
            let mut labels=vec![usize::MAX;n];
            let mut arrival=vec![f64::INFINITY;n];
            let mut owner=vec![usize::MAX;n];
            for r in 0..k {
                labels[r]=r;
                for &(j,d) in &graph[r] {if j>=k && d-weights[r]<arrival[j]{arrival[j]=d-weights[r];owner[j]=r;}}
            }
            for _ in k..n {
                let i=(k..n).filter(|&i|labels[i]==usize::MAX).min_by(|&a,&b|arrival[a].total_cmp(&arrival[b]))?;
                if !arrival[i].is_finite(){return None;}
                labels[i]=owner[i];
                for &(j,d) in &graph[i] {if labels[j]==usize::MAX && arrival[i]+d<arrival[j]{arrival[j]=arrival[i]+d;owner[j]=labels[i];}}
            }
            Some(labels)
        };
        let areas=|labels:&[usize]|{
            let mut out=vec![0.;k];
            for (i,&r) in labels.iter().enumerate(){out[r]+=cell_areas[i];}
            out
        };
        let total=boundary.unsigned_area();
        let score=|areas:&[f64]|areas.iter().zip(&self.ratios).map(|(a,t)|(a/total-t).abs()).fold(0.,f64::max);
        let mut weights=vec![0.;k];
        let labels=assign(&weights).ok_or_else(failure)?;
        let mut best_error=if connected(&labels){score(&areas(&labels))}else{f64::INFINITY};
        let mut best=if best_error.is_finite(){Some(labels)}else{None};
        // Calibrate against the connected assignment itself, not independent
        // Euclidean thresholds, whose ownership can skip over another room.
        for _ in 0..16 {for r in 0..k {
            let current=assign(&weights).ok_or_else(failure)?;
            let target=self.ratios[r]*total;
            let mut local=((areas(&current)[r]-target).abs(),weights[r]);
            let mut lo=weights.iter().copied().fold(f64::INFINITY,f64::min)-span;
            let mut hi=weights.iter().copied().fold(f64::NEG_INFINITY,f64::max)+span;
            for _ in 0..14 {
                weights[r]=(lo+hi)/2.;
                let labels=assign(&weights).ok_or_else(failure)?;
                let a=areas(&labels);
                let error=(a[r]-target).abs();
                if error<local.0 {local=(error,weights[r]);}
                let error=score(&a);
                if error<best_error && connected(&labels){best_error=error;best=Some(labels);}
                if a[r]<target {lo=weights[r];}else{hi=weights[r];}
                if best_error<=0.01 {return best.ok_or_else(failure);}
            }
            weights[r]=local.1;
        }}
        best.ok_or_else(failure)
    }
}
