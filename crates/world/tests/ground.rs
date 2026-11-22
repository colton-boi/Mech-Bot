//! Disc-against-footprint geometry for movement. Owner: the movement agent.

use world::{Prism, PrismKind, World};

/// A 10 m square from (0, 0) to (10, 10), 6 m tall.
fn square() -> Prism {
    Prism::rect(PrismKind::Building, [0.0, 0.0], [10.0, 10.0], 0.0, 6.0)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-5
}

#[test]
fn the_nearest_edge_point_is_on_the_closest_side() {
    let (q, d) = square().nearest_edge(-3.0, 4.0);
    assert!(close(q[0], 0.0) && close(q[1], 4.0) && close(d, 3.0), "{q:?} {d}");
    let (q, d) = square().nearest_edge(2.0, 5.0);
    assert!(close(q[0], 0.0) && close(d, 2.0), "inside, the west side is nearest: {q:?}");
}

#[test]
fn a_disc_overlaps_when_its_rim_crosses_an_edge() {
    let p = square();
    assert!(p.overlaps_disc(-2.0, 5.0, 2.5));
    assert!(!p.overlaps_disc(-2.5, 5.0, 2.5), "touching is not overlapping");
    assert!(p.overlaps_disc(5.0, 5.0, 2.5), "center inside");
    assert!(!p.overlaps_disc(-2.0, -2.0, 2.5), "clear of the corner (2.83 m away)");
}

#[test]
fn disc_exit_is_the_shortest_shift_out() {
    let p = square();
    let d = p.disc_exit(-1.0, 5.0, 2.5).unwrap();
    assert!(close(d[0], -1.5) && close(d[1], 0.0), "{d:?}");
    let d = p.disc_exit(1.0, 5.0, 2.5).unwrap();
    assert!(close(d[0], -3.5) && close(d[1], 0.0), "center inside: {d:?}");
    let d = p.disc_exit(0.0, 5.0, 2.5).unwrap();
    assert!(close(d[0], -2.5), "center on the edge goes out: {d:?}");
    assert!(p.disc_exit(-3.0, 5.0, 2.5).is_none());
    let d = p.disc_exit(-1.0, 5.0, 2.5).unwrap();
    assert!(!p.overlaps_disc(-1.0 + d[0], 5.0 + d[1], 2.5), "after the shift the disc is clear");
}

#[test]
fn a_hole_is_open_ground() {
    let ring = vec![[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]];
    let hole = vec![[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]];
    let p = Prism::new(PrismKind::Building, 0.0, 6.0, ring, vec![hole]);
    assert!(!p.overlaps_disc(10.0, 10.0, 2.5), "in the middle of the courtyard");
    let d = p.disc_exit(6.0, 10.0, 2.5).unwrap();
    assert!(close(d[0], 1.5), "pushed back into the courtyard: {d:?}");
}

#[test]
fn ground_under_catches_a_roof_the_disc_overlaps() {
    let mut world = World::flat(100.0);
    world.prisms.push(square());
    assert_eq!(world.ground_under(-1.0, 5.0, 2.5, 7.0), 6.0);
    assert_eq!(world.ground_under(-2.6, 5.0, 2.5, 7.0), 0.0);
    assert_eq!(world.ground_under(-1.0, 5.0, 2.5, 5.0), 0.0, "below the roof it is a wall");
}
