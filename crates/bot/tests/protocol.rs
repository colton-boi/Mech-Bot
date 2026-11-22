//! Drives the real bot binary over pipes, as the server does.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

struct Session {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Session {
    fn start() -> Session {
        Session::with_env(&[("MECH_POLICY", "example")])
    }

    fn with_env(env: &[(&str, &str)]) -> Session {
        let mut child = Command::new(env!("CARGO_BIN_EXE_bot"))
            .env_remove("MECH_POLICY")
            .env_remove("MECH_PARAMS")
            .envs(env.iter().copied())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("the bot starts");
        let (stdin, stdout) = (child.stdin.take().unwrap(), BufReader::new(child.stdout.take().unwrap()));
        Session { child, stdin, stdout }
    }

    fn send(&mut self, text: &str) {
        self.stdin.write_all(text.as_bytes()).unwrap();
        self.stdin.flush().unwrap();
    }

    fn reply(&mut self) -> Value {
        let mut line = String::new();
        self.stdout.read_line(&mut line).unwrap();
        serde_json::from_str(&line).unwrap_or_else(|e| panic!("stdout carried a non-message {line:?}: {e}"))
    }
}

fn obs(tick: u32) -> String {
    format!(
        "{}\n",
        json!({ "type": "obs", "tick": tick, "self": { "weapons": [null, null], "in_zone": true }, "zone": { "next_center": [10, 0] } })
    )
}

#[test]
fn it_survives_bad_input_answers_only_the_newest_observation_and_exits_on_end() {
    let mut bot = Session::start();
    bot.send("{\"type\":\"hello\",\"protocol\":\"mechfight/1\",\"slot\":2,\"seed\":1,\"map\":{\"dir\":\"/no/such/map\",\"files\":[]},\"config\":{}}\n");
    let ready = bot.reply();
    assert_eq!(ready["type"], "ready");
    assert!(ready["name"].as_str().unwrap().len() <= 32);

    bot.send(&obs(2));
    let first = bot.reply();
    assert_eq!(first["tick"], 2);
    assert_eq!(first["move"], json!([1.0, 0.0]), "it heads for the next circle, east");

    bot.send("this is not json\n\n{\"type\":\"weather\",\"rain\":true}\n");
    bot.send(&format!("{}{}{}", obs(4), obs(6), obs(8)));
    assert_eq!(bot.reply()["tick"], 8, "queued observations: only the newest is answered");

    bot.send("{\"type\":\"obs\",\"tick\":10,\"self\":\"garbled\"}\n");
    let coasted = bot.reply();
    assert_eq!(coasted["tick"], 10, "an observation it cannot read still gets an answer");
    assert_eq!(coasted["move"], json!([1.0, 0.0]), "it coasts on its last move");

    bot.send(&obs(12));
    assert_eq!(bot.reply()["tick"], 12, "and it recovers");

    bot.send("{\"type\":\"end\",\"place\":3,\"kills\":1}\n");
    drop(bot.stdin);
    assert!(bot.child.wait().unwrap().success());
}

const HELLO_NO_MAP: &str =
    "{\"type\":\"hello\",\"protocol\":\"mechfight/1\",\"slot\":2,\"seed\":1,\"map\":{\"dir\":\"/no/such/map\",\"files\":[]},\"config\":{}}\n";

#[test]
fn the_default_policy_is_apex_and_answers_without_a_map() {
    let mut bot = Session::with_env(&[]);
    bot.send(HELLO_NO_MAP);
    assert_eq!(bot.reply()["name"], "apex");
    for tick in [2, 4, 6] {
        bot.send(&obs(tick));
        assert_eq!(bot.reply()["tick"], tick);
    }
    drop(bot.stdin);
    assert!(bot.child.wait().unwrap().success());
}

#[test]
fn bad_params_files_and_unknown_policies_still_play() {
    let dir = std::env::temp_dir().join(format!("bot-params-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let garbled = dir.join("garbled.json");
    std::fs::write(&garbled, "{\"aggro_close_dist\": ").unwrap();
    let unknown = dir.join("unknown.json");
    std::fs::write(&unknown, "{\"no_such_param\": 1}").unwrap();
    let missing = dir.join("missing.json");
    for (policy, params) in [("aggro", &garbled), ("aggro", &unknown), ("aggro", &missing), ("no-such-policy", &garbled)] {
        let mut bot = Session::with_env(&[("MECH_POLICY", policy), ("MECH_PARAMS", params.to_str().unwrap())]);
        bot.send(HELLO_NO_MAP);
        assert_eq!(bot.reply()["type"], "ready", "{policy} {params:?}");
        bot.send(&obs(2));
        assert_eq!(bot.reply()["tick"], 2, "{policy} {params:?}");
        drop(bot.stdin);
        assert!(bot.child.wait().unwrap().success());
    }
    let good = dir.join("good.json");
    std::fs::write(&good, "{\"aggro_close_dist\": 45}").unwrap();
    let mut bot = Session::with_env(&[("MECH_POLICY", "survive"), ("MECH_PARAMS", good.to_str().unwrap())]);
    bot.send(HELLO_NO_MAP);
    assert_eq!(bot.reply()["name"], "cbot");
    bot.send(&obs(2));
    assert_eq!(bot.reply()["tick"], 2);
    drop(bot.stdin);
    assert!(bot.child.wait().unwrap().success());
    std::fs::remove_dir_all(&dir).ok();
}
