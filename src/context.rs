// SPDX-License-Identifier: GPL-3.0-only
//! The user's dictation context, as the prompt Whisper is conditioned on.
//!
//! The manifest declares `[capabilities] context = true`, so the daemon hands
//! this backend the dictation context in force on every `/v1` request, as two
//! JSON headers: `x-voice-prompt`, a string saying what the user is dictating,
//! and `x-voice-vocabulary`, an array of terms spelled the way the user wants
//! them. Each is left out when its half is empty. A Super STT daemon sends the
//! same headers as `x-stt-prompt` and `x-stt-vocabulary`; a Super Voice daemon
//! sends both spellings.
//!
//! Whisper takes such text as the transcript that came before the audio: the
//! `<|startofprev|>` tokens OpenAI's decoder fills with a caller's
//! `initial_prompt`. It follows the spellings and style of that text rather
//! than instructions in it, so OpenAI's guide prompts it with a list of names.
//! The two halves therefore fold into one text: the terms joined with commas,
//! then the prompt, the shape the OpenAI backend sends `whisper-1`.
//!
//! The terms come first because they are the half Whisper can use most
//! directly, and [`fit`] keeps the start of a context too long for the budget.

use anyhow::{Result, anyhow};
use axum::http::HeaderMap;
use tokenizers::Tokenizer;

/// The prompt's header, then its Super STT spelling.
const PROMPT: [&str; 2] = ["x-voice-prompt", "x-stt-prompt"];
/// The vocabulary's header, then its Super STT spelling.
const VOCABULARY: [&str; 2] = ["x-voice-vocabulary", "x-stt-vocabulary"];

/// The context the request carries, folded into one text (see [`fold`]).
/// Empty when the request carries none.
///
/// A header that does not decode is logged and left out rather than failing
/// the request: a malformed context is the daemon's bug, and the user's audio
/// should still be transcribed.
#[must_use]
pub fn from_headers(headers: &HeaderMap) -> String {
    let prompt: Option<String> = decode(headers, PROMPT);
    let terms: Vec<String> = decode(headers, VOCABULARY).unwrap_or_default();
    fold(prompt.as_deref(), &terms)
}

/// The first of `names` the request carries, decoded from its JSON.
fn decode<T: serde::de::DeserializeOwned>(headers: &HeaderMap, names: [&str; 2]) -> Option<T> {
    let (name, value) = names
        .iter()
        .find_map(|&name| headers.get(name).map(|value| (name, value)))?;
    match value
        .to_str()
        .map_err(|e| e.to_string())
        .and_then(|raw| serde_json::from_str(raw).map_err(|e| e.to_string()))
    {
        Ok(decoded) => Some(decoded),
        Err(e) => {
            log::warn!("ignoring the {name} header, which does not decode: {e}");
            None
        }
    }
}

/// The terms joined with commas, then a period and the prompt. Blank terms
/// and a blank prompt are left out, so no context at all is the empty string.
#[must_use]
pub fn fold(prompt: Option<&str>, terms: &[String]) -> String {
    let terms: Vec<&str> = terms
        .iter()
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .collect();
    let prompt = prompt.map(str::trim).filter(|p| !p.is_empty());
    let mut parts = Vec::with_capacity(2);
    if !terms.is_empty() {
        parts.push(terms.join(", "));
    }
    if let Some(prompt) = prompt {
        parts.push(prompt.to_string());
    }
    parts.join(". ")
}

/// `context`, trimmed, and cut to about `max_tokens` tokens if it runs longer:
/// the start of it is kept, which [`fold`] fills with the terms, and the cut
/// goes back to the last whitespace before the limit, so that a term or a word
/// is kept whole or not at all.
///
/// About, because the caller tokenizes the kept text again with the leading
/// space Whisper's prompts carry; it truncates to the limit after that.
///
/// # Errors
/// If the tokenizer fails on the context.
pub fn fit<'a>(tokenizer: &Tokenizer, context: &'a str, max_tokens: usize) -> Result<&'a str> {
    let context = context.trim();
    let encoding = tokenizer
        .encode(context, false)
        .map_err(|e| anyhow!("tokenizing the dictation context: {e}"))?;
    let offsets = encoding.get_offsets();
    let Some(&(first_dropped, _)) = offsets.get(max_tokens) else {
        return Ok(context);
    };
    let kept = cut_before(context, first_dropped);
    log::info!(
        "the dictation context is {} tokens, over the {max_tokens} Whisper is prompted with; \
         keeping its first {} of {} characters",
        offsets.len(),
        kept.chars().count(),
        context.chars().count(),
    );
    Ok(kept)
}

/// `text` up to byte `end`, cut back to the last whitespace before it when
/// there is one, and trimmed. A text with no whitespace before `end`, as
/// Chinese or Japanese is written, is cut at `end` itself.
fn cut_before(text: &str, end: usize) -> &str {
    let mut end = end.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    let head = &text[..end];
    let at_space = text[end..].starts_with(char::is_whitespace);
    let head = match head.rfind(char::is_whitespace) {
        Some(space) if !at_space => &head[..space],
        _ => head,
    };
    head.trim_end()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for &(name, value) in pairs {
            map.insert(name, value.parse().unwrap());
        }
        map
    }

    #[test]
    fn no_context_is_empty() {
        assert_eq!(from_headers(&HeaderMap::new()), "");
        assert_eq!(fold(Some("  "), &[" ".into(), String::new()]), "");
    }

    #[test]
    fn the_terms_lead_joined_with_commas_then_the_prompt() {
        let map = headers(&[
            (
                "x-voice-prompt",
                r#""An email to my team about the release.""#,
            ),
            ("x-voice-vocabulary", r#"["Super Voice", " ", "Qwen"]"#),
        ]);
        assert_eq!(
            from_headers(&map),
            "Super Voice, Qwen. An email to my team about the release."
        );
    }

    #[test]
    fn either_half_stands_alone() {
        // The daemon escapes a header to ASCII; the JSON decode undoes it.
        let prompt = headers(&[("x-voice-prompt", r#""Notas de la reuni\u00f3n""#)]);
        assert_eq!(from_headers(&prompt), "Notas de la reuni\u{f3}n");
        let vocabulary = headers(&[("x-voice-vocabulary", r#"["CubeCL", "Burn"]"#)]);
        assert_eq!(from_headers(&vocabulary), "CubeCL, Burn");
    }

    /// A Super STT daemon sends only the old names; a Super Voice one sends
    /// both, with the same values, and the new name is the one read.
    #[test]
    fn the_super_stt_names_are_read_when_the_new_ones_are_absent() {
        let old = headers(&[
            ("x-stt-prompt", r#""Notes""#),
            ("x-stt-vocabulary", r#"["Kokoro"]"#),
        ]);
        assert_eq!(from_headers(&old), "Kokoro. Notes");
        let both = headers(&[("x-voice-prompt", r#""New""#), ("x-stt-prompt", r#""Old""#)]);
        assert_eq!(from_headers(&both), "New");
    }

    /// A header that does not decode costs its half of the context, not the
    /// request.
    #[test]
    fn an_undecodable_half_is_left_out() {
        let map = headers(&[
            ("x-voice-prompt", "not json"),
            ("x-voice-vocabulary", r#"["Burn"]"#),
        ]);
        assert_eq!(from_headers(&map), "Burn");
    }

    #[test]
    fn a_cut_keeps_whole_words() {
        let text = "Siobhan, Niamh, Qwen. Notes";
        // A cut inside `Niamh` goes back to the space before it, and the
        // trailing comma is kept with the term before.
        assert_eq!(cut_before(text, 12), "Siobhan,");
        // A cut right at a space keeps everything before it.
        assert_eq!(cut_before(text, 8), "Siobhan,");
        // Text with no space before the cut is cut where it is.
        assert_eq!(
            cut_before("\u{4f60}\u{597d}\u{4e16}\u{754c}", 6),
            "\u{4f60}\u{597d}"
        );
        // A cut inside a multi-byte character moves back to its start.
        assert_eq!(cut_before("\u{4f60}\u{597d}", 4), "\u{4f60}");
    }
}
