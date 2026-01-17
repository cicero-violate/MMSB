use mmsb_judgment::JudgmentToken;

fn main() {
    let token: JudgmentToken = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
    let _clone = token.clone();
}
