---

# Fact-Checker — Design Notes

By Brian (bmvoss123@gmail.com)

> **Note on language**: these notes are written entirely in English, in the
> interest of time. Code comments and docstrings throughout the
> implementation are bilingual (Japanese first, then English), matching the
> existing repository convention — but the write-up itself is English-only.
> If Japanese had been an explicit requirement for this document, I'd have
> asked for more time, or scoped the deliverable differently to do a
> proper bilingual (or Japanese-first) write-up rather than a rushed
> translation.

## 1. What "fact-checking" means in this implementation

This implementation treats fact-checking as **evidence retrieval + narrow,
checkable grading** — not as an LLM deciding what's true from its own
knowledge. Concretely: search retrieves candidate evidence, and the LLM is
only ever asked the narrow question "does this specific evidence text support
this specific claim?" (an entailment check), never the open-ended "is this
true?" or "is this source trustworthy?" — the latter is exactly what an LLM
can't reliably do (see below).

## 2. Why web search, and the credibility problem

I leaned toward web search because it has a real chance of surfacing a
high-quality, vetted source — but an LLM isn't actually equipped to *vouch*
for a source's credibility on its own. Asking an LLM "is this a reliable
domain?" is pattern-matching on reputation it saw in training; it has no live
signal about retractions, satire, SEO spam, or a source that's simply wrong
this time.

So credibility isn't decided by the LLM at all. It's enforced by two
deterministic mechanisms instead:

- **A static domain-tiering table** (`utils.py`) — official/wire sources,
  mainstream outlets, everything else — assigned by a fixed lookup at
  retrieval time, never by LLM judgment. (This business operates in Japan, so
  Japanese official bodies and mainstream outlets — e.g. `go.jp`, `ac.jp`,
  NHK, Kyodo, Asahi, Yomiuri, Nikkei — sit at the same tiers as their
  international counterparts, not below them.)
- **Cross-domain corroboration** — a claim only counts as supported if
  independent tier-1/2 domains agree, computed in plain Python
  (`verify_subgraph.py`'s `parse_verdict`), not asserted by the LLM.

**Why SNS search specifically was not used**, given it's one of the alternatives worth
naming: social platforms are uniquely exposed to failure modes that a normal
web source isn't — coordinated influence campaigns, bandwagon effects where
genuine users repeat a trending claim without verifying it themselves, and
inauthentic/bot accounts manufacturing apparent agreement. All three produce
the *appearance* of volume or consensus without any actual independent
verification behind it. Because corroboration here is counted by distinct
*domain*, not by mention or post count, this is already structurally limited
regardless of how it entered the evidence pool: however many posts a
campaign or bot swarm produces on Facebook, X, or Reddit, they all share one
domain, and that domain sits at tier 3 — it cannot contribute a corroborating
domain no matter the volume. That's a deliberate property of counting
domains rather than counting agreement, not something bolted on
specifically for SNS.

There's a practical operations argument to be made, as well.
Querying SNS platforms directly as a primary evidence source means
integrating each platform's own API individually — and those APIs tend to
be far more rate-limited, paywalled, and quota-constrained than general web
search (X/Twitter's API pricing and Reddit's API pricing changes are the
obvious examples). A pipeline that fans out to several platform-specific
APIs per fact-check, on top of the retry loop this design already runs, would
be slower and more expensive to operate at any real volume, and would add a
maintenance burden of keeping N separate platform integrations working as
each one changes its terms. That maintenance burden isn't just infrastructure, 
but also engineer-hours: each platform-specific integration needs its own
internal documentation, auth/credential handling, and quirks explained
somewhere, which is more surface area a new engineer has to read through
and understand before they can safely touch this part of the system,
lengthening onboarding rather than shortening it. General web search already
surfaces the public, indexable slice of SNS content incidentally (Facebook
posts and Reddit threads showed up in evidence throughout live testing)
without paying that integration, rate-limit, or documentation cost, so this
design gets whatever legitimate signal exists on social platforms for free,
without taking on the operational and onboarding overhead of treating them
as a first-class source.

## 3. What I focused on in the agent design, given accuracy matters

- **The reflection loop doesn't re-ask an LLM to grade itself.** The bot's 
  `reflector` decision is deterministic: `good` only when `direction == "true"`, 
  confidence ≥ 0.7, **and** corroborating domains ≥ 2. This is a deliberate 
  deviation from the README's literal spec, where the reflector node makes its 
  own LLM call — doing that would reintroduce the exact "agent judging its own
  trustworthiness" problem this design is built to avoid. Every mistake by an LLM
  in a regulated industry is a major liability waiting to hit.

- **Direction and confidence are separate fields, not one conflated score.**
  Originally verification returned a single 0.0-1.0 "score" meant to mean
  "how strongly does evidence support this claim being true." Live testing
  caught the LLM conflating that with confidence in its own conclusion.
  This bug was exposed via a claim it correctly judged *false* having returned 
  with score=1.0, because it read "score" as confidence-in-conclusion, not
  direction-of-truth. Splitting this into an explicit `direction`
  (`"true"`/`"false"`/`"unclear"`) plus a separate `confidence` magnitude
  closes this structurally rather than just cycling about it with additional
  prompts — a categorical field can't be silently inverted the way a single 
  float can.

- **The loop never silently ships an unresolved answer.** If `MAX_LOOPS` is
  hit while still `needs_fix`, the verdict becomes `escalate` and always
  routes to a human-in-the-loop — the README's original design would finalize 
  at the loop cap regardless of evidence quality.

- **A false claim gets a decisive, citable answer instead of a shrug.**
  Verification tracks "supports" and "contradicts" as independent signals per
  piece of evidence, not just "supports." If a claim is well-corroborated as
  *false* (≥2 independent domains contradict it), the verdict becomes
  `refuted` — a distinct outcome from `needs_fix` — and the final answer
  cites the refuting sources directly, rather than looping toward a `good`
  verdict a false claim can never reach and eventually giving a vague
  non-answer. (This was tested by asking the model about something that was 
  demonstrably, obviously false.)

- **Human-in-the-loop is biased toward pausing, not the exception.** 
  Given TempestAI operates in a highly regulated industry, `human_review` is the 
  default outcome for anything that isn't unambiguously strong evidence 
  (confidence ≥ 0.9 **and** ≥ 3 corroborating domains) — auto-approval is the 
  narrow carve-out, not the common case. This is also a deviation from the 
  training example's always-pause-before-every-output pattern: here the gate 
  is conditional and tied to evidence quality, not automatic. In practice, 
  across every live test run so far, this bar was strict enough that not one 
  run auto-approved — see the observations below.

## Bugs found and fixed during live testing

- **Silent evidence accumulation without deduplication:** `evidence` accumulates
  across re-search loops; without de-duplicating by URL, the same source
  found again on a retry piled up multiple times, and the final citation list
  repeated the same sources several times over. Fixed by filtering out
  already-seen URLs in the `search` wrapper before merging.

- **Free-text rationale fed directly into search, unbounded:** The retry
  query was originally built by appending the LLM's full free-text
  explanation (truncated at a fixed character count) onto the original
  question. In one live run this actually derailed the search entirely —
  the truncated fragment caused Tavily to return results about the
  dictionary definition of the word "is." Fixed by having the LLM produce a
  short, explicit `search_suggestion` keyword phrase instead of repurposing
  its explanation as a query.

- **A silently swallowed API failure:** A local SSL certificate issue caused
  Tavily calls to fail, but the original code treated any non-"results"
  response as "0 results found" — a real failure would have been
  indistinguishable from "no evidence exists for this claim." Fixed by
  raising on an error response so it retries and surfaces properly instead
  of masquerading as a legitimate empty result.

- **The domain-tiering table didn't generalize** past the US/Japan sources it
  was seeded with. Testing "Is Ottawa the capital of Canada?" and "Is
  Mongolian an official language of the UK?" surfaced two gaps: 
    (1) a maximally clear, unanimous, zero-contradiction claim (Ottawa) still
    couldn't clear the corroboration bar because none of the actual sources a
    search engine returns for that class of question — Britannica, tourism
    boards, reference services — were recognized as anything but tier 3; 

    **and**

    (2) government ccTLD domains like `gov.uk` weren't recognized as tier 1 at
    all, because the suffix check only matched bare `.gov`/`.go.jp`, missing
    the case where "gov" is the domain's own apex label rather than a suffix.
    Fixed by adding Britannica to tier 2, and by replacing the suffix check
    with a check for "gov" appearing as its own dot-separated label — which
    catches `whitehouse.gov`, `gov.uk`, and `assets.publishing.service.gov.uk`
    uniformly, without false-positiving on something like `govtech.com`.

- **A single conflated "score" field let direction and confidence get mixed up.** 
  See the design-notes bullet above — this was caught by deliberately
  testing a demonstrably false claim, not by trying to break the system
  adversarially. Fixed by splitting into `direction` (categorical) and
  `confidence` (magnitude).

## Observation: the corroboration bar is a real constraint, not just theater

Three test questions, chosen to span "genuinely contested", "demonstrably
true", and "demonstrably false":

| Question | direction | confidence | corroborating / refuting domains | outcome |
|---|---|---|---|---|
| Is Tokyo the capital of Japan? (legal-status ambiguity is real trivia, as Tokyo is not a city itself) | true | up to 0.9 | 1 / 0 | escalate |
| Is Ottawa the capital of Canada? (uncontested) | true | 0.95 (stable across all 4 loops) | 1 / 0 | escalate |
| Is Mongolian an official language of the UK? (false) | **false** | 0.90-0.95 | 0 / 1 | escalate |

In all three cases `direction`/`confidence` were accurate and stable — the
system correctly identified truth vs. falsehood every time, including the
false claim, with a clean, correct line of reasoning and (correctly) zero 
fabricated supporting citations for it. None of the three auto-approved, and 
none reached `refuted` either — each got stuck one domain short of the 2-domain
corroboration bar in whichever direction mattered. This is intentional: 
corroboration is checked against real independent sources, not against how 
confident the LLM sounds. The Ottawa case specifically is worth its own 
discussion below, since it points at a structural limitation rather than 
just a one-off gap.

## Limitation 1: even the most basic true claims can't skip human review

Ottawa/Canada was deliberately the simplest possible test case: a fact with
no real ambiguity, near-unanimous agreement, and the highest-confidence LLM
read observed in any test (`confidence=0.95`, stable across all 4 retries).
It still escalated to human review every single time, because the
auto-approve bar requires 2 independent tier ≤ 2 corroborating domains, and
this claim only ever surfaced one (Britannica) — Wikipedia is deliberately
excluded from counting, and nothing else in the top results was recognized
above tier 3.

This is a structural consequence of the current thresholds, not a one-off
gap: as long as `good`'s auto-approve path requires 2 independent tier ≤ 2
domains, there's no query — however basic — guaranteed to skip human review.
In practice that means *every* fact-check, including the ones where the
answer was never actually in doubt, adds a human-approval step to the
pipeline. In a regulated environment, that's a real trade-off to name
explicitly: the bias toward "lean on HITL" (this is a purposeful design 
choice) carries a throughput cost, since it means slower time to serve 
customers, including the simplest ones, not just the genuinely uncertain ones.

## Limitation 2: the tiering table is deliberately non-exhaustive

It should be clarified here. The domain-tiering table was never meant to be 
a comprehensive classifier of every legitimate source on the web. It exists 
to cut down on the *simplest* cases cheaply and deterministically, not to replace 
judgment completely.

The two tiers scale differently, though:

- **Tier 1 (official/institutional) is structurally enumerable.** Government
  and academic domains follow real naming conventions — `.gov`, `.go.jp`,
  `gov.uk`, `.edu`, `.ac.jp` — so a pattern-based rule (like the "gov" label
  check added this session) keeps working on domains never seen before. This
  part reasonably approaches "solved."

- **Tier 2 (mainstream/reference) has no such pattern** and doesn't scale the
  same way. Every entry (Britannica, Asahi, NYT, ...) was hard-coded after 
  being noticed missing during testing. This pattern was acceptable for a first 
  prototype, but a full, manually-written list will always lag behind an 
  effectively unbounded set of legitimate publications, and will always be biased 
  toward whatever sources happened to be unearthed during user testing.

The corroboration mechanism is the actual backstop for sources outside this
list — an unrecognized-but-genuinely-independent source doesn't get to
rubber-stamp a claim either way, it just also doesn't count toward the 2-domain 
corroboration bar until someone adds it. Longer-term, tier 2 specifically would 
be better served by leaning more on raw independent-agreement count over 
uncategorized sources rather than a curated allowlist, or by delegating that 
curation to an already-maintained external authority (e.g. a media-literacy 
organization's outlet list, or something like Wikipedia's own perennial-sources 
list) instead of continuing to manually create entries reactively as gaps are found.

## Limitation 3: the citation list doesn't apply the same discipline as the corroboration count

`reflector`'s decision is well-protected against social-media manipulation —
see the SNS discussion in section 2 — because corroboration is counted by
domain, and an entire platform's worth of coordinated or bot-driven posts is
still just one (tier-3) domain. But that protection only covers the
*verdict*. It doesn't cover the *citations*. Citations can be formatted differently
per site, and deciding whether to treat any given link as a citation or as a link
to somewhere else for business reasons is outside of this project's scope.

`finalizer` cites every URL the LLM marked as `supports_claim` or
`contradicts_claim`, with no tier filtering at all. In live testing, Facebook
posts and Reddit threads showed up in final citation lists more than once
(e.g. the first Ottawa run cited a Facebook post from "Parliament of
Canada" alongside Britannica and Wikipedia). Those particular examples were
fine content-wise, but the mechanism has no way to tell "official
institution's Facebook page" apart from "a single viral, possibly
inauthentic post" — both get listed the same way, with no visual or
structural distinction from a `.gov` or encyclopedia citation. A single
SNS post can never *drive* a verdict on its own, but it can still ride along
in the citation list of a verdict reached some other way (e.g. via human
approval at the escalate gate), lending it unearned visual parity with
genuinely vetted sources.

Moving forward, it may be more effective to exclude tier-3 sources (or SNS
domains specifically) from the citation list entirely, or tag each citation 
with its tier in the output so a reader can see at a glance which sources are 
verified references versus unverified social posts, rather than presenting all 
citations equally.

## Example run (true-claim path)

Included alongside the false-claim run below because it's the concrete
transcript behind the "even the most basic true claims can't skip human
review" limitation discussed above — the claim is as uncontested as they
come, `direction`/`confidence` reflect that (`true`, 0.95, stable across all
4 loops), and it still escalates purely on domain count:

```
$ python fact_checker_bot.py "Is Ottawa the capital of Canada?"
--- Fact-Checker 開始 (thread_id=is-ottawa-the-capital-of-canada) ---
[2026-07-09T08:29:07] [prepare_query] 初回クエリ (initial query): Is Ottawa the capital of Canada?
[2026-07-09T08:29:08] [call_api] 5 件取得 (5 results retrieved)
[2026-07-09T08:29:08] [tag_and_extract] tiers=[2, 3, 3, 3, 3]
[2026-07-09T08:29:08] [search] 検索結果 (search results):
- [tier2] britannica.com: Ottawa | History, Facts, & Points of Interest | Britannica
- [tier3] en.wikipedia.org: Ottawa - Wikipedia
- [tier3] ottawatourism.ca: About Ottawa | Ottawa Tourism
- [tier3] brightsparktravel.ca: Capital of Canada: Why Ottawa?
- [tier3] youtube.com: Ottawa Overview | An informative introduction to Ottawa, Ontario
[2026-07-09T08:29:12] [verify] direction=true confidence=0.95 corroborating_domains=1 refuting_domains=0
[2026-07-09T08:29:12] [reflector] direction=true confidence=0.95 domains=1 refuting_domains=0 loop=1 -> needs_fix
[2026-07-09T08:29:12] [prepare_query] 再検索クエリ (retry query): Is Ottawa the capital of Canada? Ottawa capital city history
... (loops 2-3 omitted for brevity — see repo history / re-run yourself) ...
[2026-07-09T08:29:32] [verify] direction=true confidence=0.95 corroborating_domains=1 refuting_domains=0
[2026-07-09T08:29:32] [reflector] direction=true confidence=0.95 domains=1 refuting_domains=0 loop=4 -> escalate

--- 人間による確認が必要です (human review required) ---
decision=escalate direction=true confidence=0.95 corroborating_domains=1 refuting_domains=0
rationale: The evidence strongly supports the claim that Ottawa is the
capital of Canada, as multiple sources confirm its status as the capital
since the creation of the Province of Canada. The information is consistent
and reliable, indicating that no significant doubts exist over the claim's
truth.
承認しますか？ y=承認(approve) / n=却下・再検索(reject & redo): y

--- 完了 (done) ---
The evidence strongly supports the claim that Ottawa is the capital of
Canada, as multiple sources confirm its status as the capital since the
creation of the Province of Canada. [...]

根拠 (sources):
- Ottawa | History, Facts, & Points of Interest | Britannica (https://www.britannica.com/place/Ottawa)
- Ottawa - Wikipedia (https://en.wikipedia.org/wiki/Ottawa)
- About Ottawa | Ottawa Tourism (https://ottawatourism.ca/en/about-ottawa)
- Capital of Canada: Why Ottawa? (https://www.brightsparktravel.ca/blog/capital-of-canada)
- Ottawa Overview | An informative introduction to Ottawa, Ontario (https://www.youtube.com/watch?v=CFsUZiRhPcM)
- Ottawa | Geography and Cartography | Research Starters | EBSCO Research (https://www.ebsco.com/research-starters/geography-and-cartography/ottawa)
- Queen Victoria Chooses Ottawa - The Historical Society of Ottawa (https://www.historicalsocietyottawa.ca/publications/ottawa-stories/momentous-events-in-the-city-s-life/queen-victoria-chooses-ottawa)

[direction=true, confidence=0.95, corroborating_domains=1]
```

Note `corroborating_domains` never moves past 1 across all four differently-
worded retries — Britannica is the only recognized tier ≤ 2 source that ever
surfaces, and Wikipedia (present in every single result set) is deliberately
excluded from counting. This is the concrete evidence behind the throughput
trade-off named above, not a hypothetical.

## Example run (false-claim path)

Chosen over the Tokyo/Ottawa runs for this write-up because it's the clearest
demonstration of `direction`/`confidence` working correctly on a claim that's
actually false, without misfiring into a wrong "good" or fabricating a
citation:

```
$ python fact_checker_bot.py "Is Mongolian an official language of the United Kingdom?"
--- Fact-Checker 開始 (thread_id=is-mongolian-an-official-language-of-the-united-kingdom) ---
[2026-07-09T08:29:47] [prepare_query] 初回クエリ (initial query): Is Mongolian an official language of the United Kingdom?
[2026-07-09T08:29:48] [call_api] 5 件取得 (5 results retrieved)
[2026-07-09T08:29:48] [tag_and_extract] tiers=[3, 3, 3, 3, 3]
[2026-07-09T08:29:48] [search] 検索結果 (search results):
- [tier3] en.wikipedia.org: Mongolian language - Wikipedia
- [tier3] blog.languagelizard.com: Mongolian Language: Interesting Facts & Resources
- [tier3] facebook.com: Does the United Kingdom have an official language? - Facebook
- [tier3] worldmapper.org: | Mongolian Language
- [tier3] reddit.com: What languages are spoken in Mongolia? - Reddit
[2026-07-09T08:29:52] [verify] direction=false confidence=0.90 corroborating_domains=0 refuting_domains=0
[2026-07-09T08:29:52] [reflector] direction=false confidence=0.90 domains=0 refuting_domains=0 loop=1 -> needs_fix
[2026-07-09T08:29:52] [prepare_query] 再検索クエリ (retry query): Is Mongolian an official language of the United Kingdom? Mongolian language status in the United Kingdom
... (loops 2-3 omitted for brevity — see repo history / re-run yourself) ...
[2026-07-09T08:30:17] [verify] direction=false confidence=0.90 corroborating_domains=0 refuting_domains=1
[2026-07-09T08:30:17] [reflector] direction=false confidence=0.90 domains=0 refuting_domains=1 loop=4 -> escalate

--- 人間による確認が必要です (human review required) ---
decision=escalate direction=false confidence=0.90 corroborating_domains=0 refuting_domains=1
rationale: The evidence provided indicates that Mongolian is the official
language of Mongolia, not the United Kingdom. It is explicitly stated in
multiple sources that the United Kingdom does not have a constitutionally
defined official language, and that English is the main language spoken.
Therefore, the claim that Mongolian is an official language of the UK is
directly contradicted by the evidence.
承認しますか？ y=承認(approve) / n=却下・再検索(reject & redo): y

--- 完了 (done) ---
The evidence provided indicates that Mongolian is the official language of
Mongolia, not the United Kingdom. [...]

根拠 (sources):
（根拠となる情報源なし / no supporting sources found）

[direction=false, confidence=0.90, corroborating_domains=0]
```

Note: The model never fabricated a "supporting sources" citation for a claim it
correctly determined was false — the citation list is honestly empty rather
than padded with irrelevant evidence, because `finalizer` only cites
`supporting_urls`/`refuting_urls` depending on `decision`, never just
whatever happens to be in `evidence`.
