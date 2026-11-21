/// Declares [`Params`]: one named `f32` per tunable, with its default and the
/// range the optimizer may search. The field order is the vector order.
macro_rules! params {
    ($($(#[$doc:meta])* $name:ident = $default:expr, [$lo:expr, $hi:expr];)*) => {
        /// The numbers CMA-ES tunes. A tuned set is committed as these defaults.
        #[derive(Clone, Debug, PartialEq)]
        pub struct Params {
            $($(#[$doc])* pub $name: f32,)*
        }

        impl Default for Params {
            fn default() -> Self {
                Self { $($name: $default,)* }
            }
        }

        /// [`Params`] as `f64`, each read back from its shortest decimal, so a default
        /// of `0.4` is exactly the Python bot's `0.4` and thresholds compare alike.
        #[derive(Clone, Debug, PartialEq)]
        pub struct ParamsF64 {
            $(pub $name: f64,)*
        }

        impl Params {
            pub const NAMES: &'static [&'static str] = &[$(stringify!($name)),*];
            /// Inclusive `(low, high)` per field, in vector order.
            pub const BOUNDS: &'static [(f32, f32)] = &[$(($lo, $hi)),*];
            pub const LEN: usize = Self::NAMES.len();

            pub fn to_vec(&self) -> Vec<f32> {
                vec![$(self.$name),*]
            }

            /// Builds params from an optimizer's vector, clamping each value into its
            /// bounds. `None` if the length is not [`Params::LEN`] or a value is NaN.
            pub fn from_vec(v: &[f32]) -> Option<Params> {
                if v.len() != Self::LEN || v.iter().any(|x| x.is_nan()) {
                    return None;
                }
                let mut values = v.iter().zip(Self::BOUNDS).map(|(x, (lo, hi))| x.clamp(*lo, *hi));
                Some(Params { $($name: values.next()?,)* })
            }

            pub fn to_f64(&self) -> ParamsF64 {
                ParamsF64 { $($name: short(self.$name),)* }
            }
        }
    };
}

impl Params {
    /// Reads a params file, `{"field": value, ...}` (the format `tune` writes).
    /// A missing field keeps its default. An unknown field, a value that is not
    /// a number, or a NaN is an error, so a stale file is not half-applied.
    /// Values are clamped into [`Params::BOUNDS`].
    pub fn from_json(text: &str) -> Result<Params, String> {
        let value: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let map = value.as_object().ok_or("a params file holds one JSON object")?;
        let mut values = Params::default().to_vec();
        for (key, v) in map {
            let index = Params::NAMES.iter().position(|n| n == key).ok_or_else(|| format!("unknown param {key:?}"))?;
            values[index] = v.as_f64().ok_or_else(|| format!("param {key:?} is not a number"))? as f32;
        }
        Params::from_vec(&values).ok_or_else(|| "a param is NaN".into())
    }
}

/// The shortest decimal that reads back as `v`, as an `f64`.
fn short(v: f32) -> f64 {
    v.to_string().parse().unwrap_or(f64::from(v))
}

// The aggro family (`cbot.rs`) reads every field below `boost_margin`. Names and
// defaults are the Python bot's (`mech-bot/bot.py` PARAMS, plus the literals in
// its code that are worth tuning). `§4 #n` marks the STRATEGY.md section 4 row
// that names the same idea; that table's other rows describe machinery neither
// bot has, so they are not here.
params! {
    /// Example: height above a target's feet to aim at, in meters. 4.5 is its center.
    aim_height = 4.5, [0.0, 9.0];
    /// Example: seconds between jumps while travelling.
    jump_period_s = 2.0, [0.5, 10.0];
    /// Example: boost when this close to the zone's edge, in meters. 0 boosts only outside the zone.
    boost_margin = 0.0, [0.0, 200.0];

    // ---- zone ----
    /// Meters kept inside the next circle, at most.
    zone_margin = 30.0, [0.0, 120.0];
    /// The same margin as a fraction of the next radius; the smaller one applies. §4 #13.
    zone_margin_frac = 0.4, [0.05, 0.8];
    /// Seconds of slack in the "walk to the safe set in time" test.
    zone_slack_s = 10.0, [0.0, 40.0];

    // ---- health ----
    /// HP fraction that starts RETREAT (survive only). §4 #4.
    retreat_frac = 0.4, [0.1, 0.8];
    /// HP fraction that ends RETREAT. §4 #6.
    resume_frac = 0.85, [0.5, 1.0];
    /// An attacker this close stops RETREAT: fight it out.
    retreat_gives_up_dist = 25.0, [0.0, 80.0];
    /// Seconds since the last hit that keep an ongoing RETREAT going.
    retreat_memory_s = 20.0, [2.0, 40.0];
    /// While retreating, stop shooting this long after the last hit taken.
    retreat_quiet_fire_s = 2.0, [0.0, 10.0];
    /// Seconds a hit keeps its attacker as the threat.
    threat_memory_s = 6.0, [1.0, 20.0];
    /// Seconds to keep facing the last hit's bearing.
    face_threat_s = 10.0, [1.0, 30.0];
    /// Seconds to face a heard weapon or movement sound.
    alert_s = 3.0, [0.0, 15.0];

    // ---- engagement ----
    /// Visible mechs within this start FIGHT (all mechs in aggro).
    engage_range = 130.0, [30.0, 250.0];
    /// Longest range gunnery shoots at.
    fire_range = 210.0, [50.0, 300.0];
    /// Non-aggro duel band, near edge. §4 #9.
    fight_min_dist = 40.0, [5.0, 120.0];
    /// Non-aggro duel band, far edge. §4 #9.
    fight_max_dist = 110.0, [30.0, 200.0];
    /// Aggro closes to this range, then strafes.
    aggro_close_dist = 30.0, [10.0, 120.0];
    /// Keep at least this far from a lightning holder (non-aggro).
    lightning_keep_away = 55.0, [0.0, 120.0];
    /// Strafe reversal period is uniform in [min, max] seconds. §4 #34.
    strafe_s_min = 1.0, [0.2, 5.0];
    strafe_s_max = 2.4, [0.2, 5.0];
    /// Aim tolerance floor, radians; the tolerance is max(this, atan(2 m / distance)).
    aim_tol_min = 0.012, [0.002, 0.1];
    /// "Engaged" fire mode: a far mech counts as engaged when its aim is this close to us.
    aim_at_me_rad = 0.18, [0.02, 0.6];
    /// Line-of-sight tests per decision when picking a target.
    max_los_checks = 5.0, [1.0, 16.0];
    /// Fire the mortar only at targets in this range band.
    mortar_min_range = 30.0, [0.0, 120.0];
    mortar_max_range = 220.0, [60.0, 300.0];
    /// Aggro: energy above which it boosts while closing on a far target.
    chase_boost_energy = 40.0, [0.0, 100.0];

    // ---- loot and drops ----
    /// Detour for a wanted weapon. §4 #27.
    loot_detour = 35.0, [0.0, 150.0];
    /// Detour for a cube.
    cube_detour = 30.0, [0.0, 150.0];
    /// Seconds without a seen enemy before DROP and FARM are allowed.
    quiet_s = 12.0, [0.0, 60.0];
    /// No looting with a visible enemy this close.
    loot_enemy_clear = 80.0, [0.0, 250.0];
    /// Farthest drop considered.
    drop_max_dist = 170.0, [0.0, 400.0];
    /// No drop with a visible enemy this close to it. §4 #26.
    drop_enemy_clear = 60.0, [0.0, 240.0];
    /// Farm NPCs this close.
    farm_range = 75.0, [0.0, 200.0];
    /// Farm only above this HP fraction.
    farm_min_hp = 0.75, [0.0, 1.0];
    /// No farming after this match time. §4 #23.
    farm_until_t = 200.0, [0.0, 480.0];

    // ---- reflexes ----
    /// Extra meters beyond the splash radius when dodging shells. §4 #35.
    shell_margin = 5.0, [0.0, 15.0];
    /// Dodge a charging laser whose aim is within this (plus the body's angular size) of us.
    laser_dodge_rad = 0.12, [0.0, 0.6];
    /// Charging lasers farther than this are ignored.
    laser_threat_range = 280.0, [0.0, 400.0];
    /// Below this horizontal speed while trying to move counts as slow.
    stuck_speed = 1.5, [0.2, 5.0];
    /// Slow decisions in a row that trigger an unstick jet.
    stuck_decisions = 6.0, [2.0, 20.0];
    /// Seconds of the unstick jet.
    stuck_jet_s = 0.9, [0.2, 3.0];

    // ---- hold spots ----
    /// Search radius for a hold spot.
    spot_search_r = 140.0, [20.0, 250.0];
    /// Spot score weight on cover. §4 #15.
    spot_w_cover = 8.0, [0.0, 20.0];
    /// Spot score weight per meter from the next centre. §4 #16.
    spot_w_center = 0.35, [0.0, 3.0];
    /// Spot score penalty for a visible enemy, falling to 0 at 100 m. §4 #17.
    spot_w_enemy = 200.0, [0.0, 600.0];
    /// Within this of the spot counts as there (doubled while holding).
    spot_reach = 5.0, [1.0, 20.0];
    /// Look-around sweep amplitude (rad) and rate (rad/s) while holding.
    scan_amp = 0.9, [0.0, 1.8];
    scan_rate = 0.5, [0.0, 3.0];

    // ---- roof (roof strategy) ----
    roof_min_h = 6.0, [0.0, 30.0];
    roof_max_h = 42.0, [6.0, 80.0];
    roof_min_area = 150.0, [0.0, 1000.0];
    roof_max_dist = 280.0, [20.0, 600.0];
    /// Energy needed per meter of climb, plus 10.
    roof_energy_per_m = 1.8, [1.0, 4.0];
    /// Failed climbs before a roof is skipped.
    roof_max_tries = 3.0, [1.0, 10.0];

    // ---- beyond parity: switches and knobs whose defaults keep Python's behaviour ----
    // Recover (analysis/TACTICS.md probe 1), aggro only: below recover_frac HP with no
    // hit for recover_quiet_s and no mech within recover_clear_dist, stop starting fights,
    // hunting, farming and shooting until recover_exit_frac, and re-pick the hold spot.
    /// Switch, on above 0.5.
    recover_on = 0.0, [0.0, 1.0];
    recover_frac = 0.6, [0.0, 1.2];
    recover_exit_frac = 0.9, [0.5, 1.0];
    recover_clear_dist = 45.0, [0.0, 150.0];
    recover_quiet_s = 5.0, [0.0, 15.0];
    // Mode detection (STRATEGY.md 2.1): log-odds of free-for-all, +4 per mech death
    // that is not our kill before 390 s, minus a capped drift while nobody dies.
    /// Log-odds at which the match counts as a free-for-all.
    ffa_threshold = 8.0, [2.0, 16.0];
    /// Switch: in a detected free-for-all, from ffa_passive_from_t on, aggro plays like
    /// survive (fights only attackers and close mechs, fires when engaged, no hunting).
    ffa_passive_on = 0.0, [0.0, 1.0];
    ffa_passive_from_t = 265.0, [0.0, 480.0];
    // Fight range by loadout (TACTICS.md probe 3): with a switch, aggro keeps its
    // 10..aggro_close_dist band only with lightning or mortar, else these bands.
    range_by_loadout_on = 0.0, [0.0, 1.0];
    aggro_gun_min = 50.0, [10.0, 150.0];
    aggro_gun_max = 70.0, [20.0, 200.0];
    aggro_laser_min = 50.0, [10.0, 150.0];
    aggro_laser_max = 70.0, [20.0, 200.0];
    /// Gun lead: seconds added to the flight time (firing delay), and the share of
    /// the target's vertical speed to lead by. Python: 0 and 0.5.
    lead_lag_s = 0.0, [0.0, 0.3];
    lead_vz_frac = 0.5, [0.0, 1.0];
    /// Laser (STRATEGY.md 3.3): release at the minimum charge on a target whose shown
    /// HP is under this. 0 is off.
    laser_finish_hp = 0.0, [0.0, 400.0];
    /// Seconds a mech stays in the track table (last seen position) after it leaves sight.
    track_memory_s = 8.0, [1.0, 30.0];
    // Mode detection, squad side (STRATEGY.md 2.1): practice only when the log-odds fall
    // below this and a squad signature has been seen. Free-for-all otherwise.
    practice_threshold = -2.0, [-6.0, 0.0];
    // Concealed hold spots (TACTICS #12): score penalty per remembered mech that can see
    // the spot. 0 is off. Only the best two dozen spots by the other terms are ray-tested.
    spot_w_conceal = 0.0, [0.0, 200.0];
    /// Switch: while recovering, the hold spot must hide from remembered mechs (TACTICS #1).
    recover_los_break = 0.0, [0.0, 1.0];
    // Fire gates (TACTICS #4). Gun tolerance is atan(r / distance) with r = this times the
    // mech radius; 0 keeps Python's 2 m.
    fire_tol_radius_frac = 0.0, [0.0, 3.0];
    /// Switch: the gun fires only on the second on-target decision in a row at one target.
    slew_gate = 0.0, [0.0, 1.0];
    // Exposure break (TACTICS #5): this many visible mechs within fire range (rounded;
    // under 1.5 is off) send aggro to the nearest spot hidden from all of them, kept for
    // break_hold_s after the count drops. An attacker within retreat_gives_up_dist is fought.
    exposed_break_n = 0.0, [0.0, 4.0];
    break_hold_s = 2.0, [0.0, 8.0];
    break_search_r = 40.0, [10.0, 100.0];
    // Silent walk (TACTICS #12): with a mech remembered within silent_range, calm states
    // walk at this fraction of the step-sound speed, without boost or jump. 0 is off.
    silent_speed_frac = 0.0, [0.0, 1.0];
    silent_range = 60.0, [0.0, 200.0];
    // Centre brake (TACTICS #14): once the next radius is under brake_r, hold the exact
    // next centre and slow down on the approach.
    centre_brake = 0.0, [0.0, 1.0];
    brake_r = 12.0, [0.0, 40.0];
    // Target score (TACTICS #11), lowest first: w_dist per 100 m, w_hp times shown HP
    // fraction, minus w_sticky for the last target and w_facing if it aims at us.
    target_score_on = 0.0, [0.0, 1.0];
    w_dist = 1.0, [0.0, 5.0];
    w_hp = 1.0, [0.0, 5.0];
    w_sticky = 0.5, [0.0, 5.0];
    w_facing = 0.5, [0.0, 5.0];
    // Free-for-all final (TACTICS section 4): unless the mode reads practice, once the next
    // radius is at most ffa_final_r: no aggro and no hunting, a hold spot scored with
    // final_w_conceal, and fights and shots only at attackers or mechs under finish_hp.
    ffa_final_on = 0.0, [0.0, 1.0];
    ffa_final_r = 70.0, [0.0, 240.0];
    final_w_conceal = 60.0, [0.0, 200.0];
    finish_hp = 300.0, [0.0, 800.0];
    // Free-for-all early caution (live showdowns: twice dead fast charging at a bot from a
    // central spawn): unless the mode reads practice, until we hold a second weapon or
    // early_cubes cubes, and before early_until_t: no aggro, no hunting, no pull toward
    // the centre in hold spots, and fights and shots only at attackers or mechs under
    // finish_hp with no other mech within isolated_r of them.
    ffa_early_on = 0.0, [0.0, 1.0];
    early_cubes = 3.0, [0.0, 12.0];
    early_until_t = 240.0, [0.0, 480.0];
    isolated_r = 60.0, [0.0, 150.0];

    // ---- navigation ----
    /// Flow-field queue pops per decision. Python spends its leftover 10 ms on the
    /// search, about 4000 pops; more makes paths appear sooner.
    flow_pops = 4096.0, [256.0, 1000000.0];
}
