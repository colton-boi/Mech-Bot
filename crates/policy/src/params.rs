macro_rules! params {
    ($($(#[$doc:meta])* $name:ident = $default:expr, [$lo:expr, $hi:expr];)*) => {
        #[derive(Clone, Debug, PartialEq)]
        pub struct Params {
            $($(#[$doc])* pub $name: f32,)*
        }

        impl Default for Params {
            fn default() -> Self {
                Self { $($name: $default,)* }
            }
        }

        #[derive(Clone, Debug, PartialEq)]
        pub struct ParamsF64 {
            $(pub $name: f64,)*
        }

        impl Params {
            pub const NAMES: &'static [&'static str] = &[$(stringify!($name)),*];
            pub const BOUNDS: &'static [(f32, f32)] = &[$(($lo, $hi)),*];
            pub const LEN: usize = Self::NAMES.len();

            pub fn to_vec(&self) -> Vec<f32> {
                vec![$(self.$name),*]
            }

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

fn short(v: f32) -> f64 {
    v.to_string().parse().unwrap_or(f64::from(v))
}

params! {
    aim_height = 4.5, [0.0, 9.0];
    jump_period_s = 2.0, [0.5, 10.0];
    boost_margin = 0.0, [0.0, 200.0];

    zone_margin = 30.0, [0.0, 120.0];
    zone_margin_frac = 0.4, [0.05, 0.8];
    zone_slack_s = 10.0, [0.0, 40.0];

    retreat_frac = 0.4, [0.1, 0.8];
    resume_frac = 0.85, [0.5, 1.0];
    retreat_gives_up_dist = 25.0, [0.0, 80.0];
    retreat_memory_s = 20.0, [2.0, 40.0];
    retreat_quiet_fire_s = 2.0, [0.0, 10.0];
    threat_memory_s = 6.0, [1.0, 20.0];
    face_threat_s = 10.0, [1.0, 30.0];
    alert_s = 3.0, [0.0, 15.0];

    engage_range = 130.0, [30.0, 250.0];
    fire_range = 210.0, [50.0, 300.0];
    fight_min_dist = 40.0, [5.0, 120.0];
    fight_max_dist = 110.0, [30.0, 200.0];
    aggro_close_dist = 30.0, [10.0, 120.0];
    lightning_keep_away = 55.0, [0.0, 120.0];
    strafe_s_min = 1.0, [0.2, 5.0];
    strafe_s_max = 2.4, [0.2, 5.0];
    aim_tol_min = 0.012, [0.002, 0.1];
    aim_at_me_rad = 0.18, [0.02, 0.6];
    max_los_checks = 5.0, [1.0, 16.0];
    mortar_min_range = 30.0, [0.0, 120.0];
    mortar_max_range = 220.0, [60.0, 300.0];
    chase_boost_energy = 40.0, [0.0, 100.0];

    loot_detour = 35.0, [0.0, 150.0];
    cube_detour = 30.0, [0.0, 150.0];
    quiet_s = 12.0, [0.0, 60.0];
    loot_enemy_clear = 80.0, [0.0, 250.0];
    drop_max_dist = 170.0, [0.0, 400.0];
    drop_enemy_clear = 60.0, [0.0, 240.0];
    farm_range = 75.0, [0.0, 200.0];
    farm_min_hp = 0.75, [0.0, 1.0];
    farm_until_t = 200.0, [0.0, 480.0];

    shell_margin = 5.0, [0.0, 15.0];
    laser_dodge_rad = 0.12, [0.0, 0.6];
    laser_threat_range = 280.0, [0.0, 400.0];
    stuck_speed = 1.5, [0.2, 5.0];
    stuck_decisions = 6.0, [2.0, 20.0];
    stuck_jet_s = 0.9, [0.2, 3.0];

    spot_search_r = 140.0, [20.0, 250.0];
    spot_w_cover = 8.0, [0.0, 20.0];
    spot_w_center = 0.35, [0.0, 3.0];
    spot_w_enemy = 200.0, [0.0, 600.0];
    spot_reach = 5.0, [1.0, 20.0];
    scan_amp = 0.9, [0.0, 1.8];
    scan_rate = 0.5, [0.0, 3.0];

    roof_min_h = 6.0, [0.0, 30.0];
    roof_max_h = 42.0, [6.0, 80.0];
    roof_min_area = 150.0, [0.0, 1000.0];
    roof_max_dist = 280.0, [20.0, 600.0];
    roof_energy_per_m = 1.8, [1.0, 4.0];
    roof_max_tries = 3.0, [1.0, 10.0];

    recover_on = 0.0, [0.0, 1.0];
    recover_frac = 0.6, [0.0, 1.2];
    recover_exit_frac = 0.9, [0.5, 1.0];
    recover_clear_dist = 45.0, [0.0, 150.0];
    recover_quiet_s = 5.0, [0.0, 15.0];
    ffa_threshold = 8.0, [2.0, 16.0];
    ffa_passive_on = 0.0, [0.0, 1.0];
    ffa_passive_from_t = 265.0, [0.0, 480.0];
    range_by_loadout_on = 0.0, [0.0, 1.0];
    aggro_gun_min = 50.0, [10.0, 150.0];
    aggro_gun_max = 70.0, [20.0, 200.0];
    aggro_laser_min = 50.0, [10.0, 150.0];
    aggro_laser_max = 70.0, [20.0, 200.0];
    lead_lag_s = 0.0, [0.0, 0.3];
    lead_vz_frac = 0.5, [0.0, 1.0];
    laser_finish_hp = 0.0, [0.0, 400.0];
    track_memory_s = 8.0, [1.0, 30.0];
    practice_threshold = -2.0, [-6.0, 0.0];
    spot_w_conceal = 0.0, [0.0, 200.0];
    recover_los_break = 0.0, [0.0, 1.0];
    fire_tol_radius_frac = 0.0, [0.0, 3.0];
    slew_gate = 0.0, [0.0, 1.0];
    exposed_break_n = 0.0, [0.0, 4.0];
    break_hold_s = 2.0, [0.0, 8.0];
    break_search_r = 40.0, [10.0, 100.0];
    silent_speed_frac = 0.0, [0.0, 1.0];
    silent_range = 60.0, [0.0, 200.0];
    centre_brake = 0.0, [0.0, 1.0];
    brake_r = 12.0, [0.0, 40.0];
    target_score_on = 0.0, [0.0, 1.0];
    w_dist = 1.0, [0.0, 5.0];
    w_hp = 1.0, [0.0, 5.0];
    w_sticky = 0.5, [0.0, 5.0];
    w_facing = 0.5, [0.0, 5.0];
    ffa_final_on = 0.0, [0.0, 1.0];
    ffa_final_r = 70.0, [0.0, 240.0];
    final_w_conceal = 60.0, [0.0, 200.0];
    finish_hp = 300.0, [0.0, 800.0];
    ffa_early_on = 0.0, [0.0, 1.0];
    early_cubes = 3.0, [0.0, 12.0];
    early_until_t = 240.0, [0.0, 480.0];
    isolated_r = 60.0, [0.0, 150.0];

    flow_pops = 4096.0, [256.0, 1000000.0];
}
