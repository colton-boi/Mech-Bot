//! The submitted bot: JSON lines on stdin, JSON lines on stdout, logs on stderr.
//!
//! A crashed or silent bot idles for the rest of the match, so nothing here
//! may stop the loop: a line that does not parse, a map that does not load,
//! and a panic in the policy all still get an answer with the right tick.

use policy::{ApexGenome, Params, Policy};
use proto::{Action, Hello, Obs, Ready};
use serde::Deserialize;
use std::io::{Read, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Larger than the pipe buffer, so one read takes every observation that is queued.
const READ_CHUNK: usize = 1 << 20;

/// Just enough of a server line to route it.
#[derive(Deserialize)]
struct Peek {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    tick: u32,
}

fn send<T: serde::Serialize>(out: &mut impl Write, message: &T) {
    let Ok(mut line) = serde_json::to_vec(message) else {
        return;
    };
    line.push(b'\n');
    if out.write_all(&line).and_then(|()| out.flush()).is_err() {
        std::process::exit(0);
    }
}

struct Bot {
    policy: Box<dyn Policy>,
    /// The last action sent, to coast on when a decision fails.
    last: Action,
}

impl Bot {
    fn hello(&mut self, line: &[u8], out: &mut impl Write) {
        match serde_json::from_slice::<Hello>(line) {
            Ok(hello) => {
                eprintln!(
                    "bot: slot {} map {} protocol {}",
                    hello.slot, hello.map.dir, hello.protocol
                );
                if catch_unwind(AssertUnwindSafe(|| self.policy.on_hello(&hello))).is_err() {
                    eprintln!("bot: the policy panicked in on_hello");
                }
            }
            Err(e) => eprintln!("bot: hello does not parse, playing on defaults: {e}"),
        }
        let name: String = self
            .policy
            .name()
            .chars()
            .filter(|c| !c.is_control())
            .take(32)
            .collect();
        send(out, &Ready { name });
    }

    fn decide(&mut self, line: &[u8], tick: u32, out: &mut impl Write) {
        let mut action = match serde_json::from_slice::<Obs>(line) {
            Ok(obs) => {
                catch_unwind(AssertUnwindSafe(|| self.policy.act(&obs))).unwrap_or_else(|_| {
                    eprintln!("bot: the policy panicked at tick {tick}, coasting");
                    self.last.coast(tick)
                })
            }
            Err(e) => {
                eprintln!("bot: observation at tick {tick} does not parse, coasting: {e}");
                self.last.coast(tick)
            }
        };
        action.tick = tick;
        action.sanitize();
        send(out, &action);
        action.debug = None;
        self.last = action;
    }
}

/// The policy the bot plays when `MECH_POLICY` is unset or unknown.
const DEFAULT_POLICY: &str = "apex";

/// Params from the file named by `MECH_PARAMS`, else the defaults. A file that
/// cannot be read or parsed is logged and ignored: the bot always plays.
fn params() -> Params {
    let Some(path) = std::env::var_os("MECH_PARAMS").filter(|p| !p.is_empty()) else {
        return Params::default();
    };
    let read = std::fs::read_to_string(&path).map_err(|e| e.to_string());
    match read.and_then(|text| Params::from_json(&text)) {
        Ok(params) => {
            eprintln!("bot: params from {}", path.to_string_lossy());
            params
        }
        Err(e) => {
            eprintln!("bot: ignoring MECH_PARAMS {}: {e}", path.to_string_lossy());
            Params::default()
        }
    }
}

/// The apex genome from the file named by `MECH_PARAMS`, else the default, with
/// the `@key=value/...` overrides in `spec` on top. Every failure is logged and
/// falls back: the bot always plays.
fn apex_genome(spec: Option<&str>) -> ApexGenome {
    let mut genome = match std::env::var_os("MECH_PARAMS").filter(|p| !p.is_empty()) {
        None => ApexGenome::default(),
        Some(path) => match ApexGenome::load_file(std::path::Path::new(&path)) {
            Ok(genome) => {
                eprintln!("bot: apex genome from {}", path.to_string_lossy());
                genome
            }
            Err(e) => {
                eprintln!("bot: ignoring MECH_PARAMS {}: {e}", path.to_string_lossy());
                ApexGenome::default()
            }
        },
    };
    if let Some(spec) = spec {
        match genome.with_overrides(spec) {
            Ok(g) => genome = g,
            Err(e) => eprintln!("bot: ignoring apex overrides {spec:?}: {e}"),
        }
    }
    genome
}

fn build_policy(name: &str) -> Option<Box<dyn Policy>> {
    if name == "apex" {
        return Some(policy::make_apex(apex_genome(None)));
    }
    if let Some(spec) = name.strip_prefix("apex@") {
        return Some(policy::make_apex(apex_genome(Some(spec))));
    }
    policy::make(name, &params())
}

fn main() {
    let name = std::env::var("MECH_POLICY").unwrap_or_else(|_| DEFAULT_POLICY.to_owned());
    let policy = build_policy(&name).unwrap_or_else(|| {
        eprintln!("bot: no policy named {name:?}, using {DEFAULT_POLICY}");
        build_policy(DEFAULT_POLICY).expect("the default policy exists")
    });
    let mut bot = Bot {
        policy,
        last: Action::default(),
    };

    let (mut stdin, mut stdout) = (std::io::stdin().lock(), std::io::stdout().lock());
    let mut chunk = vec![0u8; READ_CHUNK];
    let mut pending: Vec<u8> = Vec::new();
    loop {
        let n = match stdin.read(&mut chunk) {
            Ok(0) => return,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                eprintln!("bot: stdin: {e}");
                return;
            }
        };
        pending.extend_from_slice(&chunk[..n]);
        let Some(end) = pending.iter().rposition(|&b| b == b'\n') else {
            continue;
        };

        let lines: Vec<(&[u8], Peek)> = pending[..end]
            .split(|&b| b == b'\n')
            .filter(|line| !line.iter().all(u8::is_ascii_whitespace))
            .filter_map(|line| match serde_json::from_slice::<Peek>(line) {
                Ok(peek) => Some((line, peek)),
                Err(e) => {
                    eprintln!("bot: skipping a line that is not a message: {e}");
                    None
                }
            })
            .collect();
        let newest_obs = lines.iter().rposition(|(_, peek)| peek.kind == "obs");
        for (i, (line, peek)) in lines.iter().enumerate() {
            match peek.kind.as_str() {
                "hello" => bot.hello(line, &mut stdout),
                "obs" if Some(i) == newest_obs => bot.decide(line, peek.tick, &mut stdout),
                "obs" => {}
                "end" => {
                    eprintln!("bot: end {}", String::from_utf8_lossy(line));
                    return;
                }
                other => eprintln!("bot: ignoring message type {other:?}"),
            }
        }
        pending.drain(..=end);
    }
}
