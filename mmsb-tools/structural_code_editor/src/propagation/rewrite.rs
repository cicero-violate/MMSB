use crate::propagation::intent::EditIntent;

pub fn rewrite_page(src: &str, intents: &[EditIntent]) -> Option<String> {
    let mut updated = src.to_string();
    let mut changed = false;

    for intent in intents {
        if let EditIntent::RenameSymbol { old, new } = intent {
            let (next, did_change) = replace_identifiers(&updated, old, new);
            if did_change {
                changed = true;
                updated = next;
            }
        }
    }

    for intent in intents {
        if let EditIntent::DeleteSymbol { name } = intent {
            let (next, did_change) = comment_out_usage(&updated, name);
            if did_change {
                changed = true;
                updated = next;
            }
        }
    }

    if changed {
        Some(updated)
    } else {
        None
    }
}

fn replace_identifiers(src: &str, old: &str, new: &str) -> (String, bool) {
    let mut out = String::with_capacity(src.len());
    let mut changed = false;
    let bytes = src.as_bytes();
    let mut idx = 0usize;

    while idx < bytes.len() {
        if is_ident_start(bytes[idx]) {
            let start = idx;
            idx += 1;
            while idx < bytes.len() && is_ident_continue(bytes[idx]) {
                idx += 1;
            }
            let token = &src[start..idx];
            if token == old {
                out.push_str(new);
                changed = true;
            } else {
                out.push_str(token);
            }
            continue;
        }

        if bytes[idx] == b'r' && idx + 2 < bytes.len() && bytes[idx + 1] == b'#' && is_ident_start(bytes[idx + 2]) {
            let start = idx;
            idx += 2;
            let ident_start = idx;
            idx += 1;
            while idx < bytes.len() && is_ident_continue(bytes[idx]) {
                idx += 1;
            }
            let token = &src[ident_start..idx];
            if token == old {
                out.push_str("r#");
                out.push_str(new);
                changed = true;
            } else {
                out.push_str(&src[start..idx]);
            }
            continue;
        }

        out.push(bytes[idx] as char);
        idx += 1;
    }

    (out, changed)
}

fn comment_out_usage(src: &str, name: &str) -> (String, bool) {
    let mut changed = false;
    let mut out = String::new();

    for line in src.lines() {
        if line.trim_start().starts_with("//") {
            out.push_str(line);
            out.push('\n');
            continue;
        }
        if line_contains_ident(line, name) {
            out.push_str("// MMSB-BROKEN ");
            out.push_str(line);
            out.push('\n');
            changed = true;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }

    if src.ends_with('\n') {
        (out, changed)
    } else {
        (out.trim_end_matches('\n').to_string(), changed)
    }
}

fn line_contains_ident(line: &str, ident: &str) -> bool {
    let bytes = line.as_bytes();
    let mut idx = 0usize;

    while idx < bytes.len() {
        if is_ident_start(bytes[idx]) {
            let start = idx;
            idx += 1;
            while idx < bytes.len() && is_ident_continue(bytes[idx]) {
                idx += 1;
            }
            if &line[start..idx] == ident {
                return true;
            }
            continue;
        }
        idx += 1;
    }

    false
}

fn is_ident_start(byte: u8) -> bool {
    byte == b'_' || (byte as char).is_ascii_alphabetic()
}

fn is_ident_continue(byte: u8) -> bool {
    byte == b'_' || (byte as char).is_ascii_alphanumeric()
}
