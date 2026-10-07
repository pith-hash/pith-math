//! Reference-vector generator for `pith-math`.
//!
//! Recomputes every vector in `tests/reference.json` from the live
//! library and either writes the file (`gen`) or verifies the committed
//! copy against the recomputation (`verify`). The file is the
//! cross-language contract: every `f64` is stored as its raw IEEE-754
//! bit pattern in 16-digit lowercase hex, so consumers never see a
//! decimal round-trip.
//!
//! Two comparison policies are recorded per vector:
//!
//! * `exact: true` — any correct implementation must reproduce the
//!   recorded bits: the result is a copied or selected element
//!   (`transpose3`, the median order statistics) or an integer-valued
//!   product whose every partial sum is exactly representable (`det3`
//!   on the standard matrix, the rotation apply, the permutation
//!   solve). Output bits must match exactly.
//! * `exact: false` — the bits are implementation-shaped: twiddle-sum
//!   kernels (FFT, DCT) depend on the platform `libm`'s `sin`/`cos`,
//!   and pivot arithmetic (`solve3`, `inverse3`, the √-scaled DCT DC
//!   block) lands an ulp or two off the mathematical answer. Different
//!   platform `libm`s may differ in the last ulp, so verification
//!   bounds the per-value error by
//!   `max(tol_abs, tol_rel · |expected|)` with the two budgets recorded
//!   in the file (hex `f64` like everything else).
//!
//! Usage:
//!
//! ```text
//! cargo run --locked --bin gen-reference -- gen   [path]   # default tests/reference.json
//! cargo run --locked --bin gen-reference -- verify [path]  # default tests/reference.json
//! ```

use pith_math::{
    Complex, dct2, dct2_2d, det3, fft, fft_real, idct2, ifft, inverse3, mat3_mul, mat3_mul_vec,
    median, solve3, transpose3,
};
use std::fs;
use std::process::ExitCode;

const DEFAULT_PATH: &str = "tests/reference.json";

/// Error budgets for the twiddle-factor kernels: one part absolute
/// (near-zero bins), one part relative to the expected magnitude. The
/// absolute floor must sit above the cancellation-noise floor of the
/// near-zero bins, where platform `libm` differences move the value by
/// a few ulps of the *summands* (1e-14 for an 8×8 block of ±3.5), not
/// ulps of the tiny result; both budgets stay orders of magnitude below
/// any real algorithmic drift.
const TOL_ABS: f64 = 1e-13;
const TOL_REL: f64 = 1e-12;

/// One recorded vector: the named kernel, its input, and the output the
/// committed file must carry.
struct Vector {
    name: &'static str,
    exact: bool,
    /// `[w, h]` for the 2D kernels; absent for 1D/scalar vectors.
    shape: Option<[usize; 2]>,
    input: Vec<f64>,
    output: Vec<f64>,
}

fn vec1(name: &'static str, exact: bool, input: Vec<f64>, output: Vec<f64>) -> Vector {
    Vector {
        name,
        exact,
        shape: None,
        input,
        output,
    }
}

/// Deterministic pseudo-input in `[-2, 2)` built from integer
/// arithmetic only — the input generation must itself be
/// platform-independent, so no PRNG and no libm.
fn lcg_input(n: usize, a: u64, m: u64, scale: f64, shift: f64) -> Vec<f64> {
    (0..n)
        .map(|i| ((i as u64 * a) % m) as f64 * scale - shift)
        .collect()
}

/// Every vector the suite shares, recomputed on each invocation.
fn compute_vectors() -> Vec<Vector> {
    let mut v: Vec<Vector> = Vec::new();

    // -- DCT -----------------------------------------------------------
    let x8: Vec<f64> = vec![0.5, -1.25, 2.0, 3.75, -0.125, 1.5, -2.5, 0.75];
    let out = dct2(&x8);
    v.push(vec1("dct2.n8", false, x8.clone(), out));
    let out = idct2(&dct2(&x8));
    v.push(vec1("idct2.roundtrip.n8", false, x8, out));

    let x32 = lcg_input(32, 37, 32, 0.125, 2.0);
    let out = dct2(&x32);
    v.push(vec1("dct2.n32", false, x32, out));

    // 8×8 block, the pHash cell: rows-then-columns separable kernel.
    let m64: Vec<f64> = (0..64)
        .map(|i| {
            let r = i / 8;
            let c = i % 8;
            ((r * 7 + c * 3) % 11) as f64 * 0.5 - 2.5
        })
        .collect();
    let mut got = m64.clone();
    dct2_2d(&mut got, 8, 8);
    v.push(Vector {
        name: "dct2.2d.8x8",
        exact: false,
        shape: Some([8, 8]),
        input: m64,
        output: got,
    });

    // Constant block: F[0] = c·√(w·h) to the last recorded bit of this
    // implementation, but the √-constant product chain is
    // implementation-shaped — compared approximately.
    let dc = vec![3.5_f64; 64];
    let mut got = dc.clone();
    dct2_2d(&mut got, 8, 8);
    v.push(Vector {
        name: "dct2.dc.8x8",
        exact: false,
        shape: Some([8, 8]),
        input: dc,
        output: got,
    });

    // -- FFT -----------------------------------------------------------
    let c8: Vec<f64> = vec![
        1.0, 0.5, -0.5, 2.0, 0.25, -1.5, 1.75, 0.0, -2.25, 0.5, 0.125, 3.0, -0.75, 1.0, 2.5, -1.0,
    ];
    let mut buf = complex(&c8);
    fft(&mut buf);
    v.push(Vector {
        name: "fft.n8",
        exact: false,
        shape: None,
        input: c8,
        output: flat(&buf),
    });

    let c64: Vec<f64> = lcg_input(64, 13, 17, 0.25, 2.0)
        .into_iter()
        .chain(lcg_input(64, 29, 19, 0.5, 4.0))
        .collect();
    let orig = complex(&c64);
    let mut buf = orig.clone();
    fft(&mut buf);
    v.push(Vector {
        name: "fft.n64",
        exact: false,
        shape: None,
        input: c64.clone(),
        output: flat(&buf),
    });

    ifft(&mut buf);
    v.push(Vector {
        name: "ifft.roundtrip.n64",
        exact: false,
        shape: None,
        input: c64,
        output: flat(&buf),
    });

    let r8 = lcg_input(8, 11, 7, 0.5, 1.5);
    let bins = fft_real(&r8);
    v.push(vec1("fft.real.n8", false, r8, flat(&bins)));

    // -- Median --------------------------------------------------------
    v.push(vec1(
        "median.even.lower_middle",
        true,
        vec![1.0, 2.0, 3.0, 4.0],
        vec![median(&mut [1.0, 2.0, 3.0, 4.0]).unwrap()],
    ));
    let lo = 1.0_f64;
    let hi = 1.0 + f64::EPSILON;
    v.push(vec1(
        "median.ulp_pair.lower_pick",
        true,
        vec![0.0, lo, hi, 2.0],
        vec![median(&mut [0.0, lo, hi, 2.0]).unwrap()],
    ));
    v.push(vec1(
        "median.pattern_member",
        true,
        vec![0.0, 0.0, 0.0, 9.0, 9.0, 9.0],
        vec![median(&mut [0.0, 0.0, 0.0, 9.0, 9.0, 9.0]).unwrap()],
    ));
    let perm: Vec<f64> = (0..63).map(|i| ((i * 37) % 63) as f64).collect();
    v.push(vec1(
        "median.permutation.63",
        true,
        perm,
        vec![
            median(
                &mut (0..63)
                    .map(|i| ((i * 37) % 63) as f64)
                    .collect::<Vec<f64>>(),
            )
            .unwrap(),
        ],
    ));
    v.push(vec1(
        "median.nan.pinned",
        true,
        vec![1.0, f64::NAN, 2.0, 3.0],
        vec![median(&mut [1.0, f64::NAN, 2.0, 3.0]).unwrap()],
    ));

    // -- solve3 / linalg -----------------------------------------------
    let a3 = [[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]];
    let b3 = [8.0, -11.0, -3.0];
    let x = solve3(&a3, &b3).expect("textbook system solves");
    // Pivot arithmetic makes the bits implementation-shaped (the answer
    // is 2, 3, −1 but the recorded f64s sit an ulp or two off) — approx.
    v.push(vec1(
        "solve3.textbook",
        false,
        flat3x3(&a3).into_iter().chain(b3).collect(),
        x.to_vec(),
    ));

    let a2 = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let x2 = solve3(&a2, &[5.0, 6.0, 7.0]).expect("permutation system solves");
    v.push(vec1(
        "solve3.permutation",
        true,
        flat3x3(&a2).into_iter().chain([5.0, 6.0, 7.0]).collect(),
        x2.to_vec(),
    ));

    let d = [[6.0, 1.0, 1.0], [4.0, -2.0, 5.0], [2.0, 8.0, 7.0]];
    v.push(vec1("det3.standard", true, flat3x3(&d), vec![det3(&d)]));

    let inv = [[2.0, 0.0, 1.0], [0.0, 3.0, 0.0], [1.0, 0.0, 2.0]];
    let got = inverse3(&inv).expect("invertible");
    // Entries like 3/5 are not representable; pivot order shapes the
    // last bits — approx.
    v.push(vec1("inverse3.sym", false, flat3x3(&inv), flat3x3(&got)));

    let p = [[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [2.0, 0.0, 1.0]];
    let q = [[2.0, 0.0, 1.0], [1.0, 3.0, 0.0], [0.0, 1.0, 2.0]];
    let mut pq = flat3x3(&p);
    pq.extend(flat3x3(&q));
    v.push(vec1("mat3.mul", true, pq, flat3x3(&mat3_mul(&p, &q))));

    let rot = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let mut rv = flat3x3(&rot);
    rv.extend([1.0, 0.0, 0.0]);
    v.push(vec1(
        "mat3.mul_vec.rot90",
        true,
        rv,
        mat3_mul_vec(&rot, &[1.0, 0.0, 0.0]).to_vec(),
    ));

    let t = [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]];
    v.push(vec1(
        "transpose3",
        true,
        flat3x3(&t),
        flat3x3(&transpose3(&t)),
    ));

    v.sort_by(|a, b| a.name.cmp(b.name));
    v
}

fn complex(flat: &[f64]) -> Vec<Complex> {
    flat.chunks_exact(2)
        .map(|c| Complex::new(c[0], c[1]))
        .collect()
}

fn flat(cs: &[Complex]) -> Vec<f64> {
    cs.iter().flat_map(|c| [c.re, c.im]).collect()
}

fn flat3x3(m: &[[f64; 3]; 3]) -> Vec<f64> {
    m.iter().flat_map(|r| r.iter().copied()).collect()
}

/// `f64` → canonical 16-digit lowercase hex of the raw bit pattern.
fn hx(v: f64) -> String {
    format!("{:016x}", v.to_bits())
}

fn push_f64s(out: &mut String, vs: &[f64], indent: &str) {
    out.push_str("[\n");
    for v in vs {
        out.push_str(indent);
        out.push('"');
        out.push_str(&hx(*v));
        out.push_str("\",\n");
    }
    // Trim the trailing comma of the last element for strict JSON.
    if !vs.is_empty() {
        let l = out.len() - 2;
        out.truncate(l);
        out.push('\n');
    }
    out.push_str(indent.trim_end());
    out.push(']');
}

/// Render the whole file byte-for-byte as it is committed.
fn render(vs: &[Vector]) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str("  \"suite\": \"pith-math\",\n");
    out.push_str(
        "  \"note\": \"every f64 is its raw IEEE-754 bit pattern as 16-digit \
lowercase hex; exact vectors must match bit-for-bit, twiddle-factor vectors \
within max(tol_abs, tol_rel * |expected|)\",\n",
    );
    out.push_str("  \"vectors\": {\n");
    for (i, v) in vs.iter().enumerate() {
        out.push_str("    \"");
        out.push_str(v.name);
        out.push_str("\": {\n");
        out.push_str("      \"exact\": ");
        out.push_str(if v.exact { "true" } else { "false" });
        out.push_str(",\n");
        if let Some([w, h]) = v.shape {
            out.push_str(&format!("      \"shape\": [{w}, {h}],\n"));
        }
        out.push_str("      \"tol_abs\": \"");
        out.push_str(&hx(TOL_ABS));
        out.push_str("\",\n");
        out.push_str("      \"tol_rel\": \"");
        out.push_str(&hx(TOL_REL));
        out.push_str("\",\n");
        out.push_str("      \"input\": ");
        push_f64s(&mut out, &v.input, "        ");
        out.push_str(",\n");
        out.push_str("      \"output\": ");
        push_f64s(&mut out, &v.output, "        ");
        out.push('\n');
        out.push_str("    }");
        if i + 1 < vs.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("  }\n}\n");
    out
}

/// Tiny JSON walker for exactly the schema `render` emits: objects,
/// arrays, strings, booleans — every numeric payload is a hex string,
/// so no float parsing ever happens.
struct Json<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Json<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            b: s.as_bytes(),
            i: 0,
        }
    }

    fn ws(&mut self) {
        while self.i < self.b.len() && self.b[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn expect(&mut self, c: u8) {
        self.ws();
        assert_eq!(self.b[self.i], c, "expected {c:?} at byte {}", self.i);
        self.i += 1;
    }

    fn peek(&mut self) -> u8 {
        self.ws();
        self.b[self.i]
    }

    fn string(&mut self) -> String {
        self.expect(b'"');
        let start = self.i;
        while self.b[self.i] != b'"' {
            self.i += 1;
        }
        let s = String::from_utf8_lossy(&self.b[start..self.i]).into_owned();
        self.i += 1;
        s
    }

    fn bool(&mut self) -> bool {
        self.ws();
        if self.b[self.i..].starts_with(b"true") {
            self.i += 4;
            true
        } else {
            assert!(
                self.b[self.i..].starts_with(b"false"),
                "bad boolean literal"
            );
            self.i += 5;
            false
        }
    }

    /// One quoted hex string → f64 (the tol_* scalars).
    fn hex_f64(&mut self) -> f64 {
        let s = self.string();
        f64::from_bits(u64::from_str_radix(&s, 16).expect("hex f64 bits"))
    }

    /// Array of hex strings → f64s.
    fn f64_array(&mut self) -> Vec<f64> {
        self.expect(b'[');
        let mut out = Vec::new();
        if self.peek() == b']' {
            self.i += 1;
            return out;
        }
        loop {
            let s = self.string();
            let bits = u64::from_str_radix(&s, 16).expect("hex f64 bits");
            out.push(f64::from_bits(bits));
            match self.peek() {
                b',' => self.i += 1,
                b']' => {
                    self.i += 1;
                    return out;
                }
                c => panic!("bad array byte {c:?}"),
            }
        }
    }
}

/// One verification failure.
struct Mismatch {
    name: String,
    detail: String,
}

/// Compare recomputed vectors against the parsed committed file.
fn verify_committed(vectors: &[Vector], committed: &str) -> Result<(), Vec<Mismatch>> {
    let mut p = Json::new(committed);
    let mut fails = Vec::new();

    // Walk to "vectors": { then read each named entry.
    p.expect(b'{');
    loop {
        let key = p.string();
        p.expect(b':');
        if key == "vectors" {
            break;
        }
        skip_value(&mut p);
        if p.peek() == b',' {
            p.i += 1;
        }
    }
    p.expect(b'{');

    let mut seen = 0;
    loop {
        match p.peek() {
            b'}' => {
                break;
            }
            b',' => {
                p.i += 1;
            }
            b'"' => {
                let name = p.string();
                p.expect(b':');
                p.expect(b'{');
                let mut exact = false;
                let mut tol_abs = (TOL_ABS, TOL_REL);
                let mut input = Vec::new();
                let mut output = Vec::new();
                loop {
                    let k = p.string();
                    p.expect(b':');
                    match k.as_str() {
                        "exact" => exact = p.bool(),
                        "shape" => skip_value(&mut p),
                        "tol_abs" => tol_abs.0 = p.hex_f64(),
                        "tol_rel" => tol_abs.1 = p.hex_f64(),
                        "input" => input = p.f64_array(),
                        "output" => output = p.f64_array(),
                        other => panic!("unexpected key {other:?}"),
                    }
                    if p.peek() == b',' {
                        p.i += 1;
                    } else {
                        p.expect(b'}');
                        break;
                    }
                }
                seen += 1;
                let Some(v) = vectors.iter().find(|v| v.name == name) else {
                    fails.push(Mismatch {
                        name,
                        detail: "committed vector has no recomputation".into(),
                    });
                    continue;
                };
                if v.exact != exact {
                    fails.push(Mismatch {
                        name: v.name.to_string(),
                        detail: format!(
                            "policy drifted: committed exact={exact}, recomputed exact={}",
                            v.exact
                        ),
                    });
                    continue;
                }
                if input.len() != v.input.len()
                    || input
                        .iter()
                        .zip(&v.input)
                        .any(|(a, b)| a.to_bits() != b.to_bits())
                {
                    fails.push(Mismatch {
                        name: v.name.to_string(),
                        detail: "input drifted".into(),
                    });
                    continue;
                }
                if output.len() != v.output.len() {
                    fails.push(Mismatch {
                        name: v.name.to_string(),
                        detail: format!(
                            "output length {} != committed {}",
                            output.len(),
                            v.output.len()
                        ),
                    });
                    continue;
                }
                for (k, (got, want)) in output.iter().zip(&v.output).enumerate() {
                    let ok = if exact {
                        got.to_bits() == want.to_bits()
                    } else {
                        let (ta, tr) = tol_abs;
                        (*got - *want).abs() <= ta.max(tr * want.abs())
                    };
                    if !ok {
                        fails.push(Mismatch {
                            name: v.name.to_string(),
                            detail: format!(
                                "output[{k}] got {:016x} want {:016x}",
                                got.to_bits(),
                                want.to_bits()
                            ),
                        });
                        break;
                    }
                }
            }
            c => panic!("bad vectors byte {c:?}"),
        }
    }
    if seen != vectors.len() {
        fails.push(Mismatch {
            name: "<suite>".to_string(),
            detail: format!(
                "committed file has {seen} vectors, recomputed {}",
                vectors.len()
            ),
        });
    }
    if fails.is_empty() { Ok(()) } else { Err(fails) }
}

/// Skip any JSON value (object/array/string/number/bool/null).
fn skip_value(p: &mut Json<'_>) {
    match p.peek() {
        b'"' => {
            let _ = p.string();
        }
        b'{' | b'[' => {
            let (open, close) = if p.peek() == b'{' {
                (b'{', b'}')
            } else {
                (b'[', b']')
            };
            let mut depth = 0usize;
            loop {
                let c = p.b[p.i];
                if c == open {
                    depth += 1;
                } else if c == close {
                    depth -= 1;
                    p.i += 1;
                    if depth == 0 {
                        return;
                    }
                }
                p.i += 1;
            }
        }
        _ => {
            while p.i < p.b.len() && !matches!(p.b[p.i], b',' | b'}' | b']') {
                p.i += 1;
            }
        }
    }
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let binding = args.next();
    let mode = match binding.as_deref() {
        Some(m @ ("gen" | "verify")) => m,
        _ => {
            eprintln!("usage: gen-reference <gen|verify> [path]");
            return ExitCode::from(2);
        }
    };
    let path = args.next().unwrap_or_else(|| DEFAULT_PATH.to_string());
    let vectors = compute_vectors();

    match mode {
        "gen" => {
            if let Err(e) = fs::write(&path, render(&vectors)) {
                eprintln!("FAIL: writing {path}: {e}");
                return ExitCode::FAILURE;
            }
            println!("wrote {} vectors to {path}", vectors.len());
            ExitCode::SUCCESS
        }
        _ => {
            let Ok(committed) = fs::read_to_string(&path) else {
                eprintln!("FAIL: cannot read {path}");
                return ExitCode::FAILURE;
            };
            match verify_committed(&vectors, &committed) {
                Ok(()) => {
                    println!(
                        "reference vectors OK: {} recomputed and verified against {path}",
                        vectors.len()
                    );
                    ExitCode::SUCCESS
                }
                Err(fails) => {
                    for f in &fails {
                        eprintln!("FAIL [{}]: {}", f.name, f.detail);
                    }
                    eprintln!("{} vector(s) drifted", fails.len());
                    ExitCode::FAILURE
                }
            }
        }
    }
}
