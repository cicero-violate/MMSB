use std::env;
use std::fs;
use std::fs::OpenOptions;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{exit, Command, Stdio};

#[derive(Debug, Clone, Copy)]
enum Task {
    Analyze,
    Report,
    Check,
}

#[derive(Debug)]
struct Config {
    task: Task,
    root: PathBuf,
    output: PathBuf,
    docs_dir: PathBuf,
    analyzer_args: Vec<String>,
}

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask must live in a subdirectory")
        .to_path_buf();

    let config = match parse_args(&root) {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("{}", err);
            print_usage();
            exit(2);
        }
    };

    match config.task {
        Task::Analyze => run_analyze(&config, &root),
        Task::Report => run_report(&config),
        Task::Check => {
            run_analyze(&config, &root);
            run_report(&config);
        }
    }
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  cargo xtask analyze [--root PATH] [--output PATH] [--skip-julia] [--verbose]");
    eprintln!("  cargo xtask report [--docs-dir PATH]");
    eprintln!("  cargo xtask check  [--root PATH] [--output PATH] [--skip-julia] [--verbose]");
}

fn parse_args(root: &Path) -> Result<Config, String> {
    let mut args = env::args().skip(1);
    let Some(cmd) = args.next() else {
        return Err("missing subcommand".to_string());
    };

    let task = match cmd.as_str() {
        "analyze" => Task::Analyze,
        "report" => Task::Report,
        "check" => Task::Check,
        _ => return Err(format!("unknown subcommand: {}", cmd)),
    };

    let mut root_path = root.to_path_buf();
    let mut output_path = root.join("docs");
    let mut docs_dir = output_path.clone();
    let mut analyzer_args = Vec::new();

    let mut pending = args.peekable();
    while let Some(arg) = pending.next() {
        match arg.as_str() {
            "--root" => {
                let Some(value) = pending.next() else {
                    return Err("missing value for --root".to_string());
                };
                root_path = resolve_path(root, value);
            }
            "--output" => {
                let Some(value) = pending.next() else {
                    return Err("missing value for --output".to_string());
                };
                output_path = resolve_path(root, value);
                docs_dir = output_path.clone();
            }
            "--docs-dir" => {
                let Some(value) = pending.next() else {
                    return Err("missing value for --docs-dir".to_string());
                };
                docs_dir = resolve_path(root, value);
            }
            "--skip-julia" | "--verbose" => {
                analyzer_args.push(arg);
            }
            _ => {
                analyzer_args.push(arg);
            }
        }
    }

    Ok(Config {
        task,
        root: root_path,
        output: output_path,
        docs_dir,
        analyzer_args,
    })
}

fn resolve_path(root: &Path, raw: String) -> PathBuf {
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        path
    } else {
        root.join(path)
    }
}

fn run_analyze(config: &Config, root: &Path) {
    if let Err(err) = fs::create_dir_all(&config.output) {
        eprintln!("failed to create output dir: {}", err);
        exit(1);
    }

    let warnings_path = config.output.join("cargo_warnings.txt");
    if let Err(err) = capture_cargo_warnings(root, &warnings_path) {
        eprintln!("failed to capture cargo warnings: {}", err);
        exit(1);
    }

    println!("Building analyzer...");
    let status = Command::new("cargo")
        .arg("build")
        .arg("--release")
        .current_dir(root)
        .status()
        .expect("failed to run cargo build");
    if !status.success() {
        exit(status.code().unwrap_or(1));
    }

    println!("Running analysis...");
    let status = Command::new("cargo")
        .arg("run")
        .arg("--release")
        .arg("--")
        .arg("--root")
        .arg(&config.root)
        .arg("--output")
        .arg(&config.output)
        .args(&config.analyzer_args)
        .current_dir(root)
        .status()
        .expect("failed to run analyzer");
    if !status.success() {
        exit(status.code().unwrap_or(1));
    }
}

fn capture_cargo_warnings(root: &Path, warnings_path: &Path) -> io::Result<()> {
    let check_status = run_and_capture(
        Command::new("cargo").arg("check").current_dir(root),
        warnings_path,
        false,
    )?;
    if check_status != 0 {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("cargo check failed with status {}", check_status),
        ));
    }

    let test_status = run_and_capture(
        Command::new("cargo")
            .arg("test")
            .arg("--no-run")
            .current_dir(root),
        warnings_path,
        true,
    )?;
    if test_status != 0 {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("cargo test failed with status {}", test_status),
        ));
    }

    Ok(())
}

fn run_and_capture(cmd: &mut Command, output: &Path, append: bool) -> io::Result<i32> {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append)
        .open(output)?;
    let stdout = file.try_clone()?;
    let status = cmd.stdout(Stdio::from(stdout)).stderr(Stdio::from(file)).status()?;
    Ok(status.code().unwrap_or(1))
}

fn run_report(config: &Config) {
    let plan_dir = config.docs_dir.join("00_refactoring_plan");
    if !plan_dir.exists() {
        eprintln!("No refactoring plan found at {:?}", plan_dir);
        eprintln!("Run 'cargo xtask analyze' first");
        exit(1);
    }

    let mut plan_files: Vec<PathBuf> = fs::read_dir(&plan_dir)
        .expect("failed to read refactoring plan directory")
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("md"))
                .unwrap_or(false)
                && path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name != "index.md")
                    .unwrap_or(true)
        })
        .collect();
    plan_files.sort();

    if plan_files.is_empty() {
        eprintln!("No refactoring plan files found in {:?}", plan_dir);
        eprintln!("Run 'cargo xtask analyze' first");
        exit(1);
    }

    let mut contents = String::new();
    for path in plan_files {
        contents.push_str(&fs::read_to_string(&path).expect("failed to read refactoring plan"));
        contents.push('\n');
    }
    let report = parse_refactoring_plan(&contents);
    print_report(&report);

    let exit_code = if report.correctness_blockers > 0 {
        2
    } else if report.layer_violations > 0 {
        1
    } else {
        0
    };
    exit(exit_code);
}

#[derive(Default)]
struct RefactoringReport {
    correctness_blockers: usize,
    layer_violations: usize,
    cluster_extractions: usize,
    cohesion_improvements: usize,
    file_renames: usize,
    phase_items: [Vec<String>; 5],
}

fn parse_refactoring_plan(contents: &str) -> RefactoringReport {
    let mut report = RefactoringReport::default();
    let mut current_phase: Option<usize> = None;

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("## Phase 1:") {
            current_phase = Some(1);
            continue;
        }
        if trimmed.starts_with("## Phase 2:") {
            current_phase = Some(2);
            continue;
        }
        if trimmed.starts_with("## Phase 3:") {
            current_phase = Some(3);
            continue;
        }
        if trimmed.starts_with("## Phase 4:") {
            current_phase = Some(4);
            continue;
        }
        if trimmed.starts_with("## Phase 5:") {
            current_phase = Some(5);
            continue;
        }
        if trimmed.starts_with("## ") {
            current_phase = None;
        }

        let Some(phase) = current_phase else {
            continue;
        };
        if !trimmed.starts_with("- ") {
            continue;
        }
        let item = trimmed.trim_start_matches("- ").trim();
        if item.is_empty()
            || item.eq_ignore_ascii_case("none.")
            || item.eq_ignore_ascii_case("none detected.")
            || item.starts_with("Verification gate")
            || item.starts_with("Extract clusters as a unit")
            || item.starts_with("Prefer creating new files")
            || item.starts_with("After each batch")
            || item.starts_with("Move lowest-layer helpers")
            || item.starts_with("Keep moves small")
            || item.starts_with("If a target module is missing")
            || item.starts_with("Prefer consolidating shared utilities")
            || item.starts_with("Avoid touching `_old/`")
            || item.starts_with("Cluster cohesion")
            || item.starts_with("Cluster cohesion")
            || item.starts_with("Phase 2 Tips")
            || item.starts_with("Phase 3 Tips")
            || item.starts_with("Phase 4 Tips")
            || item.starts_with("Phase 5 Tips")
        {
            continue;
        }

        let include = match phase {
            1 => item.starts_with("Move ") || item.contains("Correctness"),
            2 => item.starts_with("Create cluster file")
                || item.starts_with("Move `")
                || item.starts_with("Move "),
            3 => item.starts_with("Move `") || item.starts_with("`"),
            4 => item.starts_with("`") || item.starts_with("Move `"),
            5 => item.starts_with("[Rust]") || item.starts_with("Move "),
            _ => false,
        };

        if include {
            report.phase_items[phase - 1].push(item.to_string());
        }
    }

    report.correctness_blockers = report.phase_items[0].len();
    report.cluster_extractions = report.phase_items[1].len();
    report.layer_violations = report.phase_items[2].len();
    report.cohesion_improvements = report.phase_items[3].len();
    report.file_renames = report.phase_items[4].len();

    report
}

fn print_report(report: &RefactoringReport) {
    println!("REFACTORING TODO LIST\n");
    println!("SUMMARY:");
    println!("  Phase 1 (Correctness blockers): {}", report.correctness_blockers);
    println!("  Phase 2 (Cluster extractions): {}", report.cluster_extractions);
    println!("  Phase 3 (Structural constraints): {}", report.layer_violations);
    println!("  Phase 4 (Cohesion improvements): {}", report.cohesion_improvements);
    println!("  Phase 5 (Ordering & renames): {}", report.file_renames);
    println!();

    for (idx, items) in report.phase_items.iter().enumerate() {
        let title = match idx {
            0 => "PHASE 1: CORRECTNESS BLOCKERS",
            1 => "PHASE 2: CLUSTER EXTRACTION",
            2 => "PHASE 3: STRUCTURAL CONSTRAINTS",
            3 => "PHASE 4: COHESION IMPROVEMENTS",
            4 => "PHASE 5: ORDERING & RENAMES",
            _ => "PHASE",
        };
        if items.is_empty() {
            continue;
        }
        println!("{}", title);
        for (i, item) in items.iter().enumerate() {
            println!("  {}. {}", i + 1, item);
        }
        println!();
    }
}
