use world::{Prism, PrismKind, Surface, Tree, World};

/// Flat ground with a 10 m cube of building east of the origin.
fn block_world() -> World {
    let mut world = World::flat(100.0);
    world.prisms.push(Prism::rect(
        PrismKind::Building,
        [10.0, -5.0],
        [20.0, 5.0],
        0.0,
        10.0,
    ));
    world
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

#[test]
fn a_shot_stops_at_the_wall_it_faces() {
    let hit = block_world()
        .raycast([0.0, 0.0, 5.0], [30.0, 0.0, 5.0])
        .expect("the building is in the way");
    assert!(
        close(hit.t, 1.0 / 3.0) && close(hit.pos[0], 10.0),
        "{hit:?}"
    );
    assert_eq!(hit.surface, Surface::Prism(0));
}

#[test]
fn a_shot_over_the_roof_or_beside_the_wall_is_clear() {
    let world = block_world();
    assert!(world.los([0.0, 0.0, 10.5], [30.0, 0.0, 10.5]));
    assert!(world.los([0.0, 6.0, 5.0], [30.0, 6.0, 5.0]));
    assert!(!world.los([0.0, 4.0, 5.0], [30.0, 4.0, 5.0]));
}

#[test]
fn a_shot_from_above_lands_on_the_roof() {
    let hit = block_world()
        .raycast([15.0, 0.0, 30.0], [15.0, 0.0, 0.0])
        .unwrap();
    assert!(close(hit.pos[2], 10.0), "{hit:?}");
}

#[test]
fn a_shot_into_the_ground_stops_at_the_ground() {
    let hit = block_world()
        .raycast([0.0, 0.0, 5.0], [5.0, 0.0, -5.0])
        .unwrap();
    assert_eq!(hit.surface, Surface::Terrain);
    assert!(
        (hit.pos[0] - 2.5).abs() < 1e-3 && hit.pos[2].abs() < 1e-3,
        "{hit:?}"
    );
}

#[test]
fn a_segment_that_starts_inside_a_solid_hits_at_once() {
    let hit = block_world()
        .raycast([15.0, 0.0, 5.0], [40.0, 0.0, 5.0])
        .unwrap();
    assert_eq!(hit.t, 0.0);
}

#[test]
fn you_shoot_under_a_bridge_and_not_through_its_deck() {
    let mut world = World::flat(100.0);
    world.prisms.push(Prism::rect(
        PrismKind::Bridge,
        [10.0, -5.0],
        [20.0, 5.0],
        5.0,
        7.0,
    ));
    assert!(world.los([0.0, 0.0, 2.0], [30.0, 0.0, 2.0]));
    assert!(!world.los([0.0, 0.0, 6.0], [30.0, 0.0, 6.0]));
    assert!(!world.solid_at([15.0, 0.0, 2.0]));
    assert!(world.solid_at([15.0, 0.0, 6.0]));
    assert_eq!(world.ceiling_at(15.0, 0.0, 0.0), Some(5.0));
    assert_eq!(
        world.surface_at(15.0, 0.0, 1.0),
        0.0,
        "under the deck you stand on the ground"
    );
    assert_eq!(
        world.surface_at(15.0, 0.0, 8.0),
        7.0,
        "from above you stand on the deck"
    );
}

#[test]
fn a_courtyard_hole_is_outside_the_prism() {
    let ring = vec![[0.0, 0.0], [30.0, 0.0], [30.0, 30.0], [0.0, 30.0]];
    let hole = vec![[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]];
    let prism = Prism::new(PrismKind::Building, 0.0, 12.0, ring, vec![hole]);
    assert!(prism.contains([5.0, 5.0, 6.0]));
    assert!(!prism.contains([15.0, 15.0, 6.0]), "the courtyard is open");
    assert!(!prism.contains([5.0, 5.0, 12.5]), "above the roof");
    assert!(!prism.contains([35.0, 5.0, 6.0]));

    let mut world = World::flat(100.0);
    world.prisms.push(prism);
    assert!(
        world.los([15.0, 15.0, 40.0], [15.0, 15.0, 1.0]),
        "you can see down into the courtyard"
    );
    let hit = world
        .raycast([15.0, 15.0, 6.0], [40.0, 15.0, 6.0])
        .expect("the courtyard wall stops a shot leaving it");
    assert!(close(hit.pos[0], 20.0), "{hit:?}");
}

#[test]
fn a_cylinder_overhanging_a_roof_edge_rests_on_the_roof() {
    let world = block_world();
    assert_eq!(
        world.ground_under(9.0, 0.0, 2.5, 12.0),
        10.0,
        "center off the roof, rim over it"
    );
    assert_eq!(
        world.ground_under(5.0, 0.0, 2.5, 12.0),
        0.0,
        "clear of the roof"
    );
    assert_eq!(
        world.ground_under(9.0, 0.0, 2.5, 3.0),
        0.0,
        "below the roof it is a wall, not a floor"
    );
}

#[test]
fn trees_block_sight_but_not_shots() {
    let mut world = World::flat(100.0);
    world.trees.push(Tree {
        x: 20.0,
        y: 0.0,
        z: 0.0,
        height: 14.0,
        crown_r: 5.0,
        crown_base: 4.0,
        trunk_r: 0.4,
        crown_x: 20.0,
        crown_y: 0.0,
    });
    let (eye, through_trunk) = ([0.0, 0.0, 2.0], [40.0, 0.0, 2.0]);
    assert!(world.los(eye, through_trunk), "shots pass through trees");
    assert!(!world.sight(eye, through_trunk), "the trunk stops sight");
    assert!(
        world.sight([0.0, 2.0, 2.0], [40.0, 2.0, 2.0]),
        "beside the trunk and under the crown"
    );
    assert!(
        !world.sight([0.0, 1.0, 9.0], [40.0, 1.0, 9.0]),
        "ten meters of dense crown"
    );
    assert!(
        world.sight([0.0, 4.9, 9.0], [40.0, 4.9, 9.0]),
        "a little foliage at the crown's edge still sees"
    );
    assert!(
        world.sight([0.0, 0.0, 20.0], [40.0, 0.0, 20.0]),
        "over the top"
    );
}
