//! The reference-vector gate, exercised from the test suite so a drifted
//! `tests/reference.json` fails `cargo test` too, not just the CI step.
//!
//! `gen` is run against throwaway paths and compared byte-for-byte with
//! the committed file — the committed copy must be exactly what the
//! current code produces. The corruption battery then proves `verify`
//! catches every drift class the file can suffer; nothing here ever
//! rewrites the committed file.

use std::path::{Path, PathBuf};
use std::process::Command;

fn bin() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_gen-reference"));
    cmd.current_dir(env!("CARGO_MANIFEST_DIR"));
    cmd
}

fn committed() -> String {
    std::fs::read_to_string("tests/reference.json").expect("read committed file")
}

/// Write `body` to a throwaway path under the system temp dir.
fn scratch(name: &str, body: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("pith-math-gate-{name}.json"));
    std::fs::write(&p, body).expect("write scratch file");
    p
}

/// Verify a scratch path, returning (success, stderr).
fn verify(path: &Path) -> (bool, String) {
    let out = bin()
        .args(["verify", path.to_str().expect("utf-8 path")])
        .output()
        .expect("spawn gen-reference");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Byte offset of the first hex digit of the first element of the named
/// entry's `"input"` or `"output"` array.
fn first_hex_of(entry: &str, field: &str, body: &str) -> usize {
    let marker = format!("\"{entry}\": {{");
    let at = body
        .find(&marker)
        .unwrap_or_else(|| panic!("{entry} present"));
    let field_marker = format!("\"{field}\": [");
    let field_at = body[at..]
        .find(&field_marker)
        .unwrap_or_else(|| panic!("{entry}.{field} present"))
        + at
        + field_marker.len();
    let quote = body[field_at..].find('"').expect("first element quote") + field_at;
    quote + 1
}

/// Replace the first occurrence of `from` with `to`, panicking if absent.
fn patch(body: &str, from: &str, to: &str) -> String {
    assert!(body.contains(from), "patch target absent: {from:?}");
    body.replacen(from, to, 1)
}

/// Flip the last hex digit of the 16-digit word starting at `at`.
fn bump_word(body: &str, at: usize) -> String {
    let hex = &body[at..at + 16];
    let bumped = format!("{:016x}", u64::from_str_radix(hex, 16).expect("hex") + 1);
    format!("{}{}{}", &body[..at], bumped, &body[at + 16..])
}

#[test]
fn verify_accepts_the_committed_file() {
    let (ok, err) = verify(Path::new("tests/reference.json"));
    assert!(ok, "gen-reference verify failed:\n{err}");
}

/// Strip the output arrays of approx (twiddle-factor) vectors: those
/// are platform-libm samples and are only meaningful through verify's
/// tolerances. Every other byte — names, exact flags, shapes, recorded
/// budgets, inputs — must match the committed file exactly.
fn mask_approx_outputs(s: &str) -> String {
    let mut out = String::new();
    let mut exact = true;
    let mut in_approx_output = false;
    for line in s.lines() {
        let t = line.trim_end();
        let trimmed = t.trim_start();
        if trimmed.starts_with('"')
            && trimmed.contains("\": {")
            && !trimmed.starts_with("\"vectors\"")
        {
            exact = true; // reset until this entry's exact flag is seen
        }
        if trimmed.starts_with("\"exact\":") {
            exact = trimmed.contains("true");
        }
        if trimmed.starts_with("\"output\": [") {
            if exact {
                in_approx_output = false;
                out.push_str(t);
                out.push('\n');
            } else {
                in_approx_output = true;
            }
            continue;
        }
        if in_approx_output {
            if trimmed.starts_with(']') {
                in_approx_output = false;
                out.push_str(t);
                out.push('\n');
            }
            continue; // drop the element line
        }
        out.push_str(t);
        out.push('\n');
    }
    out
}

#[test]
fn regenerated_contract_matches_committed() {
    // gen → throwaway path; every libm-independent field must match the
    // committed file byte-for-byte. The approx outputs themselves are
    // platform samples: the same run must verify its own file, and the
    // committed file is checked by verify_accepts_the_committed_file.
    let tmp = std::env::temp_dir().join("pith-math-reference-regen.json");
    let out = bin()
        .args(["gen", tmp.to_str().expect("utf-8 temp path")])
        .output()
        .expect("spawn gen-reference");
    assert!(
        out.status.success(),
        "gen failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let fresh = std::fs::read_to_string(&tmp).expect("read regenerated file");
    let committed = committed();
    assert_eq!(
        mask_approx_outputs(&fresh),
        mask_approx_outputs(&committed),
        "tests/reference.json contract drifted (names/exact/shape/tols/inputs)"
    );
    let (ok, err) = verify(&tmp);
    assert!(ok, "verify of a fresh gen failed:\n{err}");
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn regenerated_file_verifies_on_its_own() {
    let tmp = std::env::temp_dir().join("pith-math-reference-regen2.json");
    let out = bin()
        .args(["gen", tmp.to_str().expect("utf-8 temp path")])
        .output()
        .expect("spawn gen-reference");
    assert!(out.status.success());
    let (ok, err) = verify(&tmp);
    assert!(ok, "verify of a fresh gen failed:\n{err}");
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn gen_into_missing_directory_fails() {
    let bad = std::env::temp_dir()
        .join("pith-math-no-such-dir")
        .join("x.json");
    let out = bin()
        .args(["gen", bad.to_str().expect("utf-8 path")])
        .output()
        .expect("spawn gen-reference");
    assert!(!out.status.success(), "gen into a missing dir must fail");
}

#[test]
fn missing_mode_is_a_usage_error() {
    let out = bin().output().expect("spawn gen-reference");
    assert_eq!(out.status.code(), Some(2), "no args must print usage");
}

#[test]
fn verify_missing_file_fails() {
    let ghost = std::env::temp_dir().join("pith-math-gate-absent.json");
    let _ = std::fs::remove_file(&ghost);
    let (ok, err) = verify(&ghost);
    assert!(!ok);
    assert!(err.contains("cannot read"), "stderr was: {err}");
}

#[test]
fn verify_rejects_garbage() {
    let p = scratch("garbage", "this is not json");
    let (ok, _) = verify(&p);
    assert!(!ok, "garbage must not verify");
    let _ = std::fs::remove_file(&p);
}

#[test]
fn verify_detects_exact_value_drift() {
    // det3.standard records -306.0 = c073200000000000; flip one bit of
    // an exact vector and verify must reject it outright.
    let body = patch(&committed(), "c073200000000000", "c073200000000001");
    let p = scratch("value-drift", &body);
    let (ok, err) = verify(&p);
    assert!(!ok, "flipped exact output must fail");
    assert!(err.contains("output[0]"), "stderr was: {err}");
    let _ = std::fs::remove_file(&p);
}

#[test]
fn verify_accepts_one_ulp_on_twiddle_vectors() {
    // fft.n8 is an approx vector: its recorded tolerance must absorb a
    // single-bit change in one output value — that is the whole point
    // of the policy, since platform libms differ in the last ulp.
    let at = first_hex_of("fft.n8", "output", &committed());
    let body = bump_word(&committed(), at);
    let p = scratch("one-ulp", &body);
    let (ok, err) = verify(&p);
    assert!(ok, "one-ulp drift on an approx vector must verify:\n{err}");
    let _ = std::fs::remove_file(&p);
}

#[test]
fn verify_detects_policy_drift() {
    let body = patch(
        &committed(),
        "\"fft.n8\": {\n      \"exact\": false",
        "\"fft.n8\": {\n      \"exact\": true",
    );
    let p = scratch("policy-drift", &body);
    let (ok, err) = verify(&p);
    assert!(!ok, "policy flip must fail");
    assert!(err.contains("policy drifted"), "stderr was: {err}");
    let _ = std::fs::remove_file(&p);
}

#[test]
fn verify_detects_input_drift() {
    // Same length, different bits: the zip comparison must catch it.
    let at = first_hex_of("dct2.n32", "input", &committed());
    let body = bump_word(&committed(), at);
    let p = scratch("input-drift", &body);
    let (ok, err) = verify(&p);
    assert!(!ok, "input drift must fail");
    assert!(err.contains("input drifted"), "stderr was: {err}");
    let _ = std::fs::remove_file(&p);
}

#[test]
fn verify_detects_output_length_drift() {
    // Drop the first output hex line of dct2.n32, keeping valid JSON.
    let body = committed();
    let hex_start = first_hex_of("dct2.n32", "output", &body) - 1; // at the quote
    let line_end = body[hex_start..].find(",\n").expect("line terminator") + hex_start + 2;
    let cut = format!("{}{}", &body[..hex_start], &body[line_end..]);
    assert_ne!(cut, body);
    let p = scratch("length-drift", &cut);
    let (ok, err) = verify(&p);
    assert!(!ok, "short output must fail");
    assert!(err.contains("output length"), "stderr was: {err}");
    let _ = std::fs::remove_file(&p);
}

#[test]
fn verify_detects_unknown_vector_name() {
    let body = patch(&committed(), "\"transpose3\"", "\"transpose3.renamed\"");
    let p = scratch("unknown-name", &body);
    let (ok, err) = verify(&p);
    assert!(!ok, "unknown name must fail");
    assert!(err.contains("no recomputation"), "stderr was: {err}");
    let _ = std::fs::remove_file(&p);
}

#[test]
fn verify_detects_missing_vector() {
    // Drop the whole transpose3 entry block (it sorts last, so no
    // dangling comma is left behind).
    let body = committed();
    let start = body.find("\"transpose3\"").expect("entry present");
    let end = body[start..].find("}\n").expect("entry end") + start + 2;
    let cut = format!("{}{}", &body[..start], &body[end..]);
    assert_ne!(cut, body);
    let p = scratch("missing-vector", &cut);
    let (ok, err) = verify(&p);
    assert!(!ok, "missing vector must fail");
    assert!(err.contains("committed file has"), "stderr was: {err}");
    let _ = std::fs::remove_file(&p);
}

#[test]
fn verify_panics_on_unexpected_entry_key() {
    let body = patch(&committed(), "\"input\": [", "\"inputs\": [");
    let p = scratch("bad-key", &body);
    let (ok, _) = verify(&p);
    assert!(!ok, "unknown entry key must abort");
    let _ = std::fs::remove_file(&p);
}

#[test]
fn verify_panics_on_malformed_array() {
    // Remove the `,\n` after transpose3's first input element: the
    // array walker then finds a quote where a separator belongs.
    let body = committed();
    let hex_start = first_hex_of("transpose3", "input", &body) - 1; // at the quote
    let sep = body[hex_start..].find(",\n").expect("separator") + hex_start;
    let cut = format!("{}{}", &body[..sep], &body[sep + 2..]);
    assert_ne!(cut, body);
    let p = scratch("malformed-array", &cut);
    let (ok, _) = verify(&p);
    assert!(!ok, "array without separator must abort");
    let _ = std::fs::remove_file(&p);
}

#[test]
fn verify_panics_on_junk_in_vectors_object() {
    let body = patch(
        &committed(),
        "  \"vectors\": {\n",
        "  \"vectors\": {\n    7,\n",
    );
    let p = scratch("junk-vectors", &body);
    let (ok, _) = verify(&p);
    assert!(!ok, "junk in the vectors object must abort");
    let _ = std::fs::remove_file(&p);
}
