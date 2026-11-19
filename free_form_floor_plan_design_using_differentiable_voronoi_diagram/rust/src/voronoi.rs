//! Forward geometry: Voronoi diagram of the sites, clipped to the boundary.
//! Two cell→site pairings live here, picked by whether a `GeosOrderHint` is
//! supplied:
//!
//! - Checkpoint path (hint present): the literal `loss.py` algorithm —
//!   positional zip + pop-on-containment. It is order-sensitive: when a clipped
//!   cell splits into a MultiPolygon, every later position shifts by one and the
//!   final piece drops out of the pairing — exactly as in Python. Used only by
//!   the fixture checkpoint tests.
//! - Standalone path (`hint` is `None` — the CLI / GIF / outcome runs): the
//!   direct voronoice site→cell mapping (each site keeps the cell it generated),
//!   which is order-independent and robust around splits. See
//!   `compute_cells_direct`.
//!
//! `GeosOrderHint` carries the two GEOS-internal orderings the Python pairing
//! depends on but that no reimplementation can recompute: the raw cell
//! iteration order and, at MultiPolygon splits, the piece iteration order
//! (given as piece areas). It is injected by the checkpoint tests only.

use geo::{Contains, Coord, LineString, MultiPolygon, Point, Polygon};
use geo_booleanop::boolean::BooleanOp;
use voronoice::{BoundingBox, Point as VPoint, VoronoiBuilder};

pub struct GeosOrderHint<'a> {
    /// raw cell position -> generating site index
    pub cell_order: &'a [usize],
    /// (raw position, piece areas in GEOS iteration order) per splitting cell
    pub split_pieces: &'a [(usize, Vec<f64>)],
}

pub struct CellGeometry {
    /// Clipped cell paired to each site, in site order. A site whose
    /// containment search finds no remaining raw cell is skipped, exactly as
    /// in Python — the list may then be shorter than the site count.
    pub cells_sorted: Vec<Polygon<f64>>,
    pub n_raw_cells: usize,
    pub n_pieces: usize,
    pub split_positions: Vec<(usize, usize)>,
}

pub fn empty_cell() -> Polygon<f64> {
    Polygon::new(LineString::new(vec![]), vec![])
}

pub fn is_empty_cell(cell: &Polygon<f64>) -> bool {
    cell.exterior().0.is_empty()
}

/// Build the voronoice diagram in a box that generously contains both the
/// boundary and all sites. The box only matters through `cell ∩ boundary`,
/// which is identical for any box ⊇ boundary.
fn build_voronoi(sites_f64: &[[f64; 2]], boundary: &Polygon<f64>) -> voronoice::Voronoi {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for c in boundary.exterior().coords() {
        min_x = min_x.min(c.x);
        min_y = min_y.min(c.y);
        max_x = max_x.max(c.x);
        max_y = max_y.max(c.y);
    }
    for &[x, y] in sites_f64 {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    let span = (max_x - min_x).max(max_y - min_y).max(1e-9);
    min_x -= span;
    min_y -= span;
    max_x += span;
    max_y += span;

    let center = VPoint {
        x: (min_x + max_x) / 2.0,
        y: (min_y + max_y) / 2.0,
    };
    VoronoiBuilder::default()
        .set_sites(
            sites_f64
                .iter()
                .map(|&[x, y]| VPoint { x, y })
                .collect::<Vec<_>>(),
        )
        .set_bounding_box(BoundingBox::new(center, max_x - min_x, max_y - min_y))
        .build()
        .expect("voronoi construction failed")
}

/// Voronoi cell polygon per site (site order).
pub fn raw_cells_per_site(sites_f64: &[[f64; 2]], boundary: &Polygon<f64>) -> Vec<Polygon<f64>> {
    let mut cells = Vec::with_capacity(sites_f64.len());
    visit_raw_rings(sites_f64, boundary, None, |_, ring| {
        cells.push(raw_cell_polygon(ring));
    });
    cells
}

#[cfg(test)]
thread_local! {
    pub(crate) static RAW_POLYGONS_BUILT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(crate) fn affected_raw_cells(
    sites: &[[f64; 2]],
    boundary: &Polygon<f64>,
    base_cells: &[Polygon<f64>],
    base_vertices: &VertexWelder,
    moved: usize,
) -> Vec<(usize, Polygon<f64>)> {
    let mut affected = Vec::new();
    visit_raw_rings(sites, boundary, Some(base_vertices), |site, ring| {
        let base = &base_cells[site].exterior().0;
        let open_base = &base[..base.len().saturating_sub(1)];
        if site == moved || ring != open_base {
            affected.push((site, raw_cell_polygon(ring)));
        }
    });
    affected
}

fn visit_raw_rings(
    sites: &[[f64; 2]],
    boundary: &Polygon<f64>,
    base: Option<&VertexWelder>,
    mut visit: impl FnMut(usize, &[Coord<f64>]),
) {
    let diagram = build_voronoi(sites, boundary);
    let mut welder = VertexWelder::with_capacity(diagram.vertices().len());
    let mut coordinates = vec![None; diagram.vertices().len()];
    let mut ring = Vec::new();
    // Visit sites in the original order, but weld each shared global vertex
    // only once before clipping. Cocircular twin IDs still pass through weld.
    for (site, cell) in diagram.iter_cells().enumerate() {
        ring.clear();
        for &id in cell.triangles() {
            let point = *coordinates[id].get_or_insert_with(|| {
                let p = &diagram.vertices()[id];
                let (x, y) = welder.point(p.x, p.y, base);
                Coord { x, y }
            });
            if ring.last() != Some(&point) {
                ring.push(point);
            }
        }
        while ring.len() > 1 && ring.first() == ring.last() {
            ring.pop();
        }
        if ring.len() < 3 {
            ring.clear();
        }
        visit(site, &ring);
    }
}

fn raw_cell_polygon(ring: &[Coord<f64>]) -> Polygon<f64> {
    #[cfg(test)]
    RAW_POLYGONS_BUILT.set(RAW_POLYGONS_BUILT.get() + 1);
    let mut closed = Vec::with_capacity(ring.len() + usize::from(!ring.is_empty()));
    closed.extend_from_slice(ring);
    closed.extend(ring.first().copied());
    Polygon::new(LineString::new(closed), vec![])
}

/// Voronoi neighbor site indices per site (the sites sharing a Voronoi edge).
/// Stable under a tiny site perturbation, so the local-gradient path caches
/// this once per step and reuses it for all 2N perturbations.
pub fn site_neighbors(sites_f64: &[[f64; 2]], boundary: &Polygon<f64>) -> Vec<Vec<usize>> {
    let voronoi = build_voronoi(sites_f64, boundary);
    let mut neighbors = vec![Vec::new(); sites_f64.len()];
    for cell in voronoi.iter_cells() {
        neighbors[cell.site()] = cell.iter_neighbors().collect();
    }
    neighbors
}

/// Clip one raw Voronoi cell to the boundary, returning the chosen piece and
/// the number of intersection pieces. On a MultiPolygon split, keep the piece
/// containing the site, or — if the site has drifted outside the boundary —
/// the largest piece (see `compute_cells_direct`).
pub fn clip_cell(raw: &Polygon<f64>, boundary: &Polygon<f64>, site: [f64; 2]) -> (Polygon<f64>, usize) {
    use geo::Area;
    let inter: MultiPolygon<f64> = raw.intersection(boundary);
    let pieces: Vec<Polygon<f64>> = inter.0;
    let n = pieces.len();
    let cell = match n {
        0 => empty_cell(),
        1 => pieces.into_iter().next().unwrap(),
        _ => {
            let s = Point::new(site[0], site[1]);
            pieces
                .into_iter()
                .max_by(|a, b| {
                    let key = |p: &Polygon<f64>| (p.contains(&s), p.unsigned_area());
                    key(a).partial_cmp(&key(b)).unwrap()
                })
                .unwrap_or_else(empty_cell)
        }
    };
    (cell, n)
}

/// True iff the raw Voronoi cell lies entirely inside the boundary, so clipping
/// it is the identity (`cell ∩ boundary == cell`) and the boolean intersection
/// can be skipped. SOUND but conservative: it may return `false` for a cell that
/// is in fact interior (a missed skip — never wrong), but it never returns
/// `true` for a cell the boundary actually cuts. Used only by the local demo
/// gradient path (`grad_local`); the exact CLI path keeps `clip_cell`.
pub(crate) fn cell_inside_boundary(raw: &Polygon<f64>, boundary: &Polygon<f64>) -> bool {
    let pts = &raw.exterior().0;
    if pts.len() < 3 {
        return false;
    }
    // cell bounding box
    let (mut cx0, mut cy0, mut cx1, mut cy1) =
        (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for c in pts {
        cx0 = cx0.min(c.x);
        cy0 = cy0.min(c.y);
        cx1 = cx1.max(c.x);
        cy1 = cy1.max(c.y);
    }
    // If any boundary edge's bbox overlaps the cell bbox, the boundary may cut
    // through the cell — bail to the real clip (conservative: never a false
    // accept, only a missed skip).
    for e in boundary.exterior().lines() {
        let (ex0, ex1) = (e.start.x.min(e.end.x), e.start.x.max(e.end.x));
        let (ey0, ey1) = (e.start.y.min(e.end.y), e.start.y.max(e.end.y));
        if cx0 <= ex1 && cx1 >= ex0 && cy0 <= ey1 && cy1 >= ey0 {
            return false;
        }
    }
    // No boundary edge is near the cell, so it is wholly inside or wholly
    // outside; one point-in-polygon settles which.
    boundary.contains(&Point::new(pts[0].x, pts[0].y))
}

/// Welds floating-point-twin vertices across the clipped pieces.
///
/// Where a shared Voronoi edge crosses the boundary, the seam intersection
/// point is computed once per neighboring cell with the segment endpoints in
/// opposite order, so the two results can differ in the last bits. Those
/// near-duplicate vertices keep the room unions from dissolving internal
/// edges. Snapping to a canonical representative within 1e-12 (nine orders
/// below any real feature size; displacement is invisible at the f32 loss
/// precision) restores the bitwise-shared edges that GEOS gets for free from
/// its single noding step.
#[derive(Default)]
pub(crate) struct VertexWelder {
    // Sorted buckets retain the old (bucket x, bucket y, encounter) lookup
    // order without hashing nine empty neighboring buckets per vertex.
    points:Vec<((i64,i64),(f64,f64))>,
    exact:Vec<((u64,u64),(f64,f64))>,
}
const WELD_TOLERANCE:f64=1e-12;
impl VertexWelder {
    fn with_capacity(n:usize)->Self{
        Self{points:Vec::with_capacity(n),exact:Vec::with_capacity(n)}
    }
    pub(crate) fn for_cells(cells:&[Polygon<f64>])->Self{
        let mut out=Self::default();
        // Base cells already carry their canonical identities. Seed their
        // actual coordinates without selecting new representatives.
        for p in cells.iter().flat_map(|c|c.exterior().0.iter()){
            let point=(p.x,p.y);let key=(p.x.to_bits(),p.y.to_bits());
            if let Err(at)=out.exact.binary_search_by_key(&key,|&(k,_)|k){
                out.exact.insert(at,(key,point));
                out.insert(point);
            }
        }
        out
    }
    fn bucket(x:f64,y:f64)->(i64,i64){((x/WELD_TOLERANCE).round() as i64,(y/WELD_TOLERANCE).round() as i64)}
    fn insert(&mut self,point:(f64,f64)){
        let key=Self::bucket(point.0,point.1);
        let at=self.points.partition_point(|&(bucket,_)|bucket<=key);
        self.points.insert(at,(key,point));
    }
    fn nearby(&self,x:f64,y:f64)->Option<(f64,f64)>{
        let (kx,ky)=Self::bucket(x,y);
        let start=self.points.partition_point(|&((bx,_),_)|bx<kx-1);
        for &((bx,by),(px,py)) in &self.points[start..]{
            if bx>kx+1{break;}
            if (by-ky).abs()<=1 && (px-x).abs()<=WELD_TOLERANCE && (py-y).abs()<=WELD_TOLERANCE{return Some((px,py));}
        }
        None
    }
    fn point(&mut self,x:f64,y:f64,base:Option<&Self>)->(f64,f64){
        let key=(x.to_bits(),y.to_bits());
        // Most raw vertices are identical across one-site perturbations.
        // Their immutable base identity needs no new memo or bucket entry.
        if let Some(p)=base.and_then(|b|b.exact.binary_search_by_key(&key,|&(k,_)|k).ok().map(|i|b.exact[i].1)){return p;}
        let at=match self.exact.binary_search_by_key(&key,|&(k,_)|k){Ok(i)=>return self.exact[i].1,Err(at)=>at};
        let existing=base.and_then(|b|b.nearby(x,y)).or_else(||self.nearby(x,y));
        let point=existing.unwrap_or_else(||{self.insert((x,y));(x,y)});
        self.exact.insert(at,(key,point));
        point
    }
}

pub(crate) fn snap_cells(cells:&mut[Polygon<f64>]){
    snap_cells_using(cells,&mut VertexWelder::default(),None);
}

pub(crate) fn snap_cells_using(cells:&mut[Polygon<f64>],welder:&mut VertexWelder,base:Option<&VertexWelder>){
    for cell in cells.iter_mut() {
        if is_empty_cell(cell) {
            continue;
        }
        let pts = &cell.exterior().0;
        let open = &pts[..pts.len() - 1];
        let mut snapped: Vec<(f64, f64)> = Vec::with_capacity(open.len());
        for c in open {
            let p = welder.point(c.x,c.y,base);
            if snapped.last() != Some(&p) {
                snapped.push(p);
            }
        }
        while snapped.len() > 1 && snapped.first() == snapped.last() {
            snapped.pop();
        }
        *cell = if snapped.len() < 3 {
            empty_cell()
        } else {
            Polygon::new(LineString::from(snapped), vec![])
        };
    }
}

/// Reorders intersection pieces to match the GEOS piece order recorded in the
/// fixture (greedy nearest-area matching; areas at a split differ by orders
/// of magnitude, so the matching is unambiguous).
fn order_pieces_by_area(mut pieces: Vec<Polygon<f64>>, areas: &[f64]) -> Vec<Polygon<f64>> {
    use geo::Area;
    let mut ordered = Vec::with_capacity(pieces.len());
    for &target in areas {
        if pieces.is_empty() {
            break;
        }
        let (best, _) = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| (i, (p.unsigned_area() - target).abs()))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        ordered.push(pieces.remove(best));
    }
    ordered.extend(pieces);
    ordered
}

pub fn compute_cells(
    sites: &[[f32; 2]],
    boundary: &Polygon<f64>,
    hint: Option<&GeosOrderHint>,
) -> CellGeometry {
    let n = sites.len();
    let sites_f64: Vec<[f64; 2]> = sites.iter().map(|&[x, y]| [x as f64, y as f64]).collect();

    let cell_of_site = raw_cells_per_site(&sites_f64, boundary);

    // Standalone runs (GIF/CLI/outcome loop) use the direct voronoice
    // site->cell mapping, which is robust and order-independent. The
    // order-dependent zip-pop pairing below exists ONLY to bit-replicate
    // loss.py's GEOS-order-dependent piece dropping for the fixture checkpoint
    // (where the hint is injected); without the hint that re-pairing could
    // mis-assign cells around a MultiPolygon split (shape_b: 10/40 sites).
    if hint.is_none() {
        return compute_cells_direct(&sites_f64, boundary, &cell_of_site);
    }

    let identity: Vec<usize> = (0..n).collect();
    let order: &[usize] = hint.map(|h| h.cell_order).unwrap_or(&identity);

    // clip cells to the boundary in raw order, expanding MultiPolygon splits
    // positionally — `raws` is NOT expanded, mirroring loss.py
    let mut cells: Vec<Polygon<f64>> = Vec::with_capacity(n + 4);
    let mut raws: Vec<&Polygon<f64>> = Vec::with_capacity(n);
    let mut split_positions = Vec::new();
    for (pos, &site_idx) in order.iter().enumerate() {
        let raw = &cell_of_site[site_idx];
        // Martinez-Rueda boolean op: pure f64, preserves uncut vertices
        // bitwise — neighboring clipped cells keep bitwise-shared edges, so
        // the room unions downstream weld without slivers.
        let inter: MultiPolygon<f64> = raw.intersection(boundary);
        let mut pieces: Vec<Polygon<f64>> = inter.0;
        match pieces.len() {
            0 => cells.push(empty_cell()),
            1 => cells.push(pieces.pop().unwrap()),
            k => {
                split_positions.push((pos, k));
                if let Some(h) = hint {
                    if let Some((_, areas)) =
                        h.split_pieces.iter().find(|(p, _)| *p == pos)
                    {
                        pieces = order_pieces_by_area(pieces, areas);
                    }
                }
                cells.extend(pieces);
            }
        }
        raws.push(raw);
    }
    let n_pieces = cells.len();

    snap_cells(&mut cells);

    // literal zip-pop pairing: first remaining (cell, raw) whose raw cell
    // contains the site claims that positional pair
    let mut rem_cells = cells;
    let mut rem_raws = raws;
    let mut cells_sorted = Vec::with_capacity(n);
    for &[x, y] in &sites_f64 {
        let pt = Point::new(x, y);
        let len = rem_cells.len().min(rem_raws.len());
        if let Some(ci) = (0..len).find(|&ci| rem_raws[ci].contains(&pt)) {
            cells_sorted.push(rem_cells.remove(ci));
            rem_raws.remove(ci);
        }
    }

    CellGeometry {
        cells_sorted,
        n_raw_cells: cell_of_site.len(),
        n_pieces,
        split_positions,
    }
}

/// Direct voronoice site->cell pairing (no hint): each site i is the generator
/// of `cell_of_site[i]`, so we clip that cell and keep it for site i. At a
/// MultiPolygon split we keep the piece that geometrically contains the site;
/// if the site has drifted OUTSIDE the boundary (so no clipped piece contains
/// it — legitimate mid-optimization) we keep the LARGEST piece instead, so the
/// cell still covers its boundary region. Dropping it to empty there would
/// leave a hole at a concave boundary notch (shape_duck ~iter 11). No
/// containment re-search, so the assignment cannot drift around a split.
fn compute_cells_direct(
    sites_f64: &[[f64; 2]],
    boundary: &Polygon<f64>,
    cell_of_site: &[Polygon<f64>],
) -> CellGeometry {
    let n = sites_f64.len();
    let mut cells_sorted: Vec<Polygon<f64>> = Vec::with_capacity(n);
    let mut n_pieces = 0usize;
    let mut split_positions = Vec::new();
    for (i, &site) in sites_f64.iter().enumerate() {
        // prefer the piece containing the site; else (site outside boundary)
        // the largest piece — see `clip_cell`.
        let (cell, np) = clip_cell(&cell_of_site[i], boundary, site);
        n_pieces += np.max(1);
        if np > 1 {
            split_positions.push((i, np));
        }
        cells_sorted.push(cell);
    }

    snap_cells(&mut cells_sorted);

    CellGeometry {
        cells_sorted,
        n_raw_cells: cell_of_site.len(),
        n_pieces,
        split_positions,
    }
}

/// Render-only cells: every piece of each site's Voronoi cell ∩ boundary, tagged
/// with the site index. Unlike `compute_cells` (which keeps ONE piece per site
/// for the bit-exact loss pairing), this keeps ALL pieces, so the rendered plan
/// covers the whole boundary — matching the Python renderer (`generator.py`,
/// which iterates `cell.geoms` for a MultiPolygon). The loss/gradient path is
/// unaffected.
pub fn render_cells(sites: &[[f32; 2]], boundary: &Polygon<f64>) -> Vec<(usize, Polygon<f64>)> {
    let sites_f64: Vec<[f64; 2]> = sites.iter().map(|&[x, y]| [x as f64, y as f64]).collect();
    let raw = raw_cells_per_site(&sites_f64, boundary);
    let mut out = Vec::with_capacity(raw.len());
    for (i, raw_cell) in raw.iter().enumerate() {
        // keep EVERY piece of this cell's intersection with the boundary, so the
        // pieces of all cells tile the boundary with no uncovered gap.
        let inter: MultiPolygon<f64> = raw_cell.intersection(boundary);
        for piece in inter.0 {
            if !is_empty_cell(&piece) {
                out.push((i, piece));
            }
        }
    }
    out
}

#[cfg(test)]
mod render_tests {
    use super::*;
    use crate::{init, shapes};
    use geo::Area;

    #[test]
    fn render_cells_covers_boundary_at_multipart_split() {
        // A small site count makes some Voronoi cells large enough to straddle a
        // concave notch of the boundary, so the cell ∩ boundary splits into more
        // than one piece. The config is chosen deterministically (init only, no
        // optimization), so — unlike the old version, which advanced the optimizer
        // to *find* this degeneracy — it does not depend on a chaotic trajectory
        // and cannot be perturbed away by a loss or init change.
        let boundary = shapes::by_name("shape_b").unwrap().polygon();
        let barea = boundary.unsigned_area();
        let sites = init::initialize_sites(&boundary, 16, 5);

        // the one-piece compute_cells path drops the smaller split piece -> a gap...
        let geom = compute_cells(&sites, &boundary, None);
        let one_piece: f64 = geom.cells_sorted.iter().map(|c| c.unsigned_area()).sum();
        assert!(barea - one_piece > 1e-3, "one-piece gap should be present: {}", barea - one_piece);

        // ...but render_cells keeps ALL pieces -> covers the whole boundary.
        let pieces = render_cells(&sites, &boundary);
        let all: f64 = pieces.iter().map(|(_, p)| p.unsigned_area()).sum();
        assert!((barea - all).abs() < 1e-5, "render_cells must cover the boundary; gap = {}", barea - all);
        assert!(pieces.len() > geom.cells_sorted.len(), "the split must expand the piece count: {} vs {}", pieces.len(), geom.cells_sorted.len());
    }
}

#[cfg(test)]
mod interior_tests {
    use super::*;
    use crate::{init, shapes};
    use geo::Area;

    #[test]
    fn affected_raw_cells_follow_a_delaunay_diagonal_flip() {
        let boundary = Polygon::new(LineString::from(vec![
            (-1., -1.), (1., -1.), (1., 1.), (-1., 1.),
        ]), vec![]);
        let sites = [[-0.4, -0.4], [0.4, -0.4], [0.4, 0.4], [-0.4, 0.4]];
        let base = raw_cells_per_site(&sites, &boundary);
        let weld = VertexWelder::for_cells(&base);
        let mut neighbors = Vec::new();
        let coordinates = |cell: &Polygon<f64>| {
            cell.exterior().0.iter().map(|p| [p.x.to_bits(), p.y.to_bits()]).collect::<Vec<_>>()
        };
        for offset in [-1e-6, 1e-6] {
            let mut moved = sites;
            moved[0][0] += offset;
            let mut links = site_neighbors(&moved, &boundary);
            links.iter_mut().for_each(|row| row.sort_unstable());
            neighbors.push(links);

            let mut full = Vec::new();
            visit_raw_rings(&moved, &boundary, Some(&weld), |_, ring| {
                full.push(raw_cell_polygon(ring));
            });
            let expected: Vec<_> = full.iter().enumerate()
                .filter(|&(i, raw)| i == 0 || *raw != base[i])
                .map(|(i, raw)| (i, coordinates(raw)))
                .collect();
            let actual: Vec<_> = affected_raw_cells(&moved, &boundary, &base, &weld, 0)
                .iter().map(|(i, raw)| (*i, coordinates(raw))).collect();
            assert_eq!(actual, expected);
        }
        assert_ne!(neighbors[0], neighbors[1], "the two perturbations must flip a real diagonal");
    }

    #[test]
    fn finite_differences_reuse_cell_workspace() {
        use crate::grad_local::{LocalGradContext, CELL_WORKSPACES_BUILT};
        use crate::loss::LossWeights;

        let boundary = shapes::by_name("shape_e").unwrap().polygon();
        let sites = init::initialize_sites(&boundary, 32, 4971);
        let labels = init::kmeans_labels(&sites, 6, 4971);
        let weights = LossWeights { w_area: 1., w_wall: 0.1, w_jog: 0.01, ..Default::default() };
        let targets = [boundary.unsigned_area() / 6.; 6];
        let context = LocalGradContext::new(&sites, &boundary, &targets, &labels, &weights);
        CELL_WORKSPACES_BUILT.set(0);
        let gradients = context.gradients(&sites);
        assert_eq!(CELL_WORKSPACES_BUILT.get(), 1, "use one cell workspace per serial gradient batch");
        let mut probe = sites.clone();
        for (i, original) in sites.iter().enumerate() {
            for axis in 0..2 {
                probe[i][axis] = original[axis] + 1e-6;
                let plus = context.local_total(&probe, i);
                probe[i][axis] = original[axis] - 1e-6;
                let minus = context.local_total(&probe, i);
                probe[i][axis] = original[axis];
                assert_eq!(gradients[i][axis].to_bits(), ((plus - minus) / (2. * 1e-6)).to_bits(),
                    "restore the workspace before the next perturbation");
            }
        }
    }

    // KR1: `cell_inside_boundary` is a SOUND trivial-accept predicate — every
    // cell it calls "inside" must have clip-identity (the boolean clip leaves the
    // area unchanged), and both classes must occur on real geometry (it is not
    // vacuously all-true or all-false). The sound direction is the safety
    // property: a `true` that is not actually clip-identity would let the caller
    // skip a clip the boundary needed, corrupting the cell.
    #[test]
    fn cell_inside_boundary_is_sound_and_nonvacuous() {
        let boundary = shapes::by_name("shape_a").unwrap().polygon();
        let sites = init::initialize_sites(&boundary, 40, 777);
        let sites_f64: Vec<[f64; 2]> = sites.iter().map(|&[x, y]| [x as f64, y as f64]).collect();
        let raw = raw_cells_per_site(&sites_f64, &boundary);

        let (mut n_true, mut n_false) = (0usize, 0usize);
        for (i, rc) in raw.iter().enumerate() {
            let pred = cell_inside_boundary(rc, &boundary);
            // reference truth: clipping is identity iff the cell is fully inside
            let (clipped, _) = clip_cell(rc, &boundary, sites_f64[i]);
            let ra = rc.unsigned_area();
            let ca = clipped.unsigned_area();
            let identity = ra > 0.0 && ((ra - ca).abs() / ra < 1e-9);
            if pred {
                n_true += 1;
                assert!(
                    identity,
                    "cell {i}: predicate said inside but clip changed area (raw {ra}, clipped {ca})"
                );
            } else {
                n_false += 1;
            }
        }
        assert!(n_true > 0, "predicate never returned true — it recognizes no interior cells");
        assert!(n_false > 0, "predicate never returned false — it recognizes no boundary cells");
    }
}
