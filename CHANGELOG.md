# Changelog

## v1.1

Release candidate `v1.1-rc1`, 2026-10-17, compared against tag `v1.0`. Two parts: the
completeness fill (this section) and the prawns neutral sentences and full-text joint
prompt (the next).

### The completeness fill

Bank and task changes under AMENDMENT_V1_1_COMPLETENESS (2026-10-17), on top of the prawns
neutral sentences and full-text joint prompt described in the next section.

- **Prawns reference case** (`comparand`, 2 texts in `banks/duty_texts_v1_1.jsonl`):
  "Reference case (for comparison only): " + the other leg's separate case with
  "Alaric" replaced by "Another man" + " Case to rate: " + this leg's separate case. The
  rule reproduces the study's banked reference cases for pairs 1, 3, 4, 5 and 7 byte for
  byte (pair 5's are checked in the tests). Asked BLAME; chat frame only.
- **Prawns duty CAUSE question** (`banks/duty_questions_v1_1.json`): "{item} How much did
  {agent} cause the illness, on a scale from 0 to 9, where 0 means not at all and 9 means
  entirely? Answer with a single digit." One question for both legs; its harm noun occurs
  in neither leg's case. Asked of the separate cases; chat frame only.
- **The stated gap uses the full-text prompt where it exists.** Where a model has
  full-text side-by-side cells for a pair, `duty_stated_gap` is their two-order mean, or
  missing with the reason if they did not read; it never falls back to the v1.0 prompt,
  so a pooled figure never mixes the two definitions across models. Elsewhere it is the
  v1.0 prompt's. `stated_prompt` records which; the v1.0 prompt's reading is always in
  `duty_stated_gap_summary`. On v1.0 data nothing changes.
- **The excuse test reads SUPPLIER and single roles:** `-T roles=` (SUPPLIER when named,
  or `-T roles=all`).
- **Single cells, both tests:** `-T cells=` takes sample ids or a JSON file of them, and
  refuses any id the selection does not render.
- **Gemini 3.1 Pro on the excuse test:** `models/gemini_pro.json` gains `bicycle` and
  `scenarios` wires identical to its duty wire (sampled, 8,192 tokens, the vendor's
  default reasoning, google-vertex, 10 draws).
- **Tests:** new `tests/test_v1_1_fill.py` (39 checks), including a refused-cell check
  made against a selection that is otherwise read, so it tests the cell check rather than
  the empty-selection guard, and a check that the stated gap has one row per model and
  pair.

### Prawns neutral sentences and a full-text prawns joint prompt

**Banks.** The prawns pair was built before the duty test had neutral sentences, so in
v1.0 it has none: it has no neutral baseline, it contributes no deviations to a model's
wording floor, and it is classed against a floor measured on the other five pairs. v1.1
adds three neutral sentences for it, written under the same rules as the other pairs'
(within three words of the pair's convention sentence; no evaluative, frequency, care,
duty or risk language; no word found in only one leg's case; no restatement of the
harm), each placed before the separate case on both legs, as on the other pairs:

1. *A framed print of a harbour hung on the wall opposite the window, beside a shelf of
   old paperbacks.*
2. *It was a Friday evening, and it had rained for much of the day before the sky
   cleared at around six o'clock.*
3. *There was quiet music in the background, and the talk turned to holidays and a film
   some of them had seen.*

**The full-text joint prompt.** The prawns pair's side-by-side (joint) prompt is a
summary of its two cases, where the other five pairs set out both cases in full. v1.1
adds a second prawns joint prompt built as theirs are: *"In two separate incidents, "*,
the professional's separate case, the private person's separate case, then the banked
question and answer form, in both orders. Each case is the banked separate case word for
word except its subject, which is written as the other pairs write theirs (*"a chef at a
restaurant served…"*, *"A man served…"*; the rule reproduces pair 5's banked joint
prompts byte for byte). The labels are CHEF and HOME_COOK, from `duty_pairs.json`, as for
the other pairs; the v1.0 prompt uses CHEF and HOME. It is read with
`-T conditions=joint_full`, and has its own view, `duty_stated_gap_full` (see above for
how the stated gap now uses it).

The eight texts are in a new file, `banks/duty_texts_v1_1.jsonl`. No v1.0 file is
changed. `banks/MANIFEST.json` gains a `versions` block listing each version's files
(v1.0: 66 duty texts; v1.1: 76, with the reference case above).

**Effect on results.** In a v1.1 read the prawns deviations join each model's pooled
wording floor (18 deviations rather than 15), so a model's floor, and the classes and
shares that depend on it, can change on every pair, not only on prawns.

**The duty task.**
- `-T bank=1.0|1.1`: the default is 1.1. Under `-T bank=1.0` every prompt is rendered
  byte for byte as before; the digests of all 1,386 v1.0 prompts, recorded before v1.1
  was made, are a test fixture (`tests/fixtures/v1_0_prompt_sha256.json`).
- `-T conditions=`: a kind (`separate`, `comparison`, `stated`, `neutral`), a
  group (`joint`, `joint_full`), or a single condition (`bare`, `cmp_pro_first`,
  `cmp_priv_first`, `cmp_full_pro_first`, `cmp_full_priv_first`, `stated`, `neutral_1`
  to `neutral_3`). For example, `-T pairs=prawns -T conditions=neutral,joint_full`
  reads the 8 new cells alone.
- An unknown pair or condition, or a selection with no samples, is refused. `-T pairs=1`
  (which Inspect parses as an integer) now works.
- The docstring no longer says prawns has no neutral sentences.

**Both tasks.** `-T prompt_check=FILE`: a pre-flight check. Every prompt the read would
send is hashed (`harness.frames.prompt_sha16`, the form the study hashed its prompts in)
and the read is refused before any call if a digest is not in the file.

**Database.** Each duty cell records its `bank_version`; cells from older logs count as
1.0. `db.build_from_readings` loads the texts of the latest version its cells were read
under, so a build from v1.0 readings is unchanged. The text rows record which bank file
they came from.

**Tests.** New `tests/test_bank_v1_1.py` (58 checks): the texts and their tampering;
the neutrals' position on both legs for a sampled and a log-probability model and on
the worksheet; the full-text joint prompt (the template rule, both orders, selection,
the parties scored the right way round, kept out of the stated gap); selection; v1.0
byte identity; the pre-flight; and the database. The
end-to-end test also builds the same read under bank 1.0, with its v1.0 count.
`test_local` and the dry-run check compare the study's worksheet prompts on the v1.0
bank, which is what that fixture holds. The dry run now reads 300 duty-test worksheet
cells (270 under v1.0).

## v1.0

The excuse test and the duty test as published with the paper: tag `v1.0`, commit
`ba96e18`. `tests/fixtures/v1_0_prompt_sha256.json` holds the digest of every prompt it
renders (1,386), and `-T bank=1.0` in v1.1 renders them byte for byte.
