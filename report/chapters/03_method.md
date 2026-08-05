# 3. Method

This chapter describes the reward pipeline stage by stage, the design decision taken at each stage, and the alternative that was rejected there. The alternatives are drawn from the deviation log and are presented in the body of the chapter rather than confined to an appendix, because the choices and their rejected competitors are as much a part of the method as the pipeline itself. The chapter closes with the pre-registration: the hypotheses, analyses, and interpretation rules were fixed in writing before any data was seen, and every departure from them was recorded on the day it occurred.

## 3.1 Overview of the pipeline

The reward is computed deterministically from recovered argument structure. There is no learned judge and no second language model in the loop. A question produces a reasoning trace, the trace is turned into a directed graph of argumentative relations, six properties of that graph are measured, and those measurements are combined with answer correctness into a single scalar reward.

```
BBH question -> policy model generates a trace          train/grpo_train.py
             -> trace split into numbered steps         reward/xaif_build.py
             -> every step pair classified              reward/ari.py
             -> directed graph, nodes steps, edges relations   reward/features.py
             -> six measurements, each in [0,1]         reward/features.py
             -> weighted sum, length-penalised, added to correctness   reward/scorer.py
```

Each stage is a single module, and the boundaries are drawn so that each can be tested in isolation. The relation model in the middle is a fixed, pretrained checkpoint that is never trained; only the policy model on the left is optimised. Because the signal is recovered rather than learned, it is inspectable at every stage: for any trace, the steps, the links between them with their confidences, and the resulting measurements can all be read off directly. That property is the point of the design, and it shapes several of the decisions below.

## 3.2 Splitting the trace into steps

The relation model adds links between statements; it does not produce the statements. In the oAMF framework [@gemechuOpenArgumentMining2025] a separate segmentation module performs that task, but those modules are trained on essays and debate transcripts [@lawrenceArgumentMiningSurvey2019], a different register from a reasoning trace. The pipeline therefore splits the trace itself: it divides on newlines first, since that is how models write chains of thought, falls back to sentence splitting for single-paragraph traces, strips numbering, and discards fragments shorter than fifteen characters.

The alternative considered here (D-001) was to use an oAMF segmentation module. It was rejected because a chain of thought is already written in discrete steps and splits reliably on newlines, so an essay-trained segmenter would add a second source of error without adding information. Doing the split directly is a methodological improvement rather than a shortcut: any structure that is subsequently measured is attributable to one model's errors instead of two models' errors compounding, which is what makes the parser-fidelity study of Chapter 4 interpretable.

A native reasoning model presents its chain of thought inside `<think>` tags, with the final answer after the closing tag. The extraction step (`_split_think`) therefore mines the reasoning from inside the tags and reads the answer from after them. This behaviour is backward compatible: a trace with no tags is treated exactly as before, which is what keeps the instruction-tuned baseline corpus and the reasoning-model corpus directly comparable. The move to a reasoning model is discussed in full in section 3.10.

## 3.3 Identifying relations

The relation model is `raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L`, the checkpoint behind the oAMF relation module and itself the argument-relation-identification model of [@ruizdolzTransformerBasedModels2021]. It takes two statements and returns one of three labels with a confidence, mapped to the Argument Interchange Format [@chesnevarArgumentInterchangeFormat2006]: Inference to RA (the first statement supports the second), Conflict to CA (the two contradict), and Rephrase to MA (the two restate each other).

In practice the checkpoint exposes a fourth class, No-Relation, and the pipeline records it separately. This allows "the model judged the pair unrelated" to be counted apart from "the model proposed a relation but below its confidence floor". The two are different events, and keeping them distinct is a small but genuine contribution to how these models should be used and reported.

The confidence floors, 0.9 for Inference and 0.7 for Conflict and Rephrase, are taken from the published implementation of the module (its checkpoint configuration) rather than from the oAMF paper, which does not state them. Pairs below the floor produce no link. These thresholds are not tuned. Adjusting a supervisor's published thresholds and then reporting an improvement would make any result impossible to defend, so they are held fixed by policy. Chapter 5 reports a sweep of these floors, but that is a robustness check on the null result rather than an attempt to find a better operating point, and it is labelled as such wherever it appears.

The alternative considered here (D-002) was to call the module through the framework's routing rather than loading the checkpoint directly. This was rejected because local routing deploys modules through Docker, which the Colab environment does not provide, and a network call for every step pair would make training unaffordable and introduce a runtime dependency on an external service. Loading the public checkpoint directly gives the same weights and the same behaviour without the container, and the pairing and thresholding logic, including the concatenation of the two statements into a single input string, is reimplemented to match the published module.

The direction of a support edge required an explicit decision, taken during the setup checks. In these checks the checkpoint was observed to return support edges pointing from conclusion to premise, the essay convention where the claim is stated first and its support follows; this orientation is not documented in the oAMF paper and is reported here as a measured property of the checkpoint rather than an attribution. A chain of thought is the reverse shape: premises accumulate and the conclusion arrives last. Support edges are therefore oriented from premise to conclusion, that is from the earlier step to the later one. This reorientation is a concrete, citable instance of the essay-to-chain-of-thought domain shift, and it is reported in the body of the method rather than buried in a code comment. A verification requirement was also set: a sample of traces scored by this implementation is to be compared against the published web service, and the agreement reported. Where that check has not been completed it is stated as such in the Limitations rather than left implied.

## 3.4 The pairing decision and the compute budget

The relation model scores pairs, one forward pass each, so the number of pairs sets the cost. Scoring every pair of an *n*-step trace is *n*(*n*−1)/2 comparisons; scoring only adjacent steps is *n*−1. For a fifteen-step trace that is 105 comparisons against fourteen. Only the all-pairs setting can detect a step that supports the conclusion from several steps earlier, which is exactly what the connectivity and support-depth measurements are designed to capture; the neighbours-only setting cannot see such a link at any price.

This trade-off was measured rather than assumed. On an L4 graphics processor the implementation scores roughly 600 pairs per second all-pairs and roughly 460 neighbours-only, the latter being overhead-dominated on small inputs and therefore a conservative floor. Projected forward, this is roughly forty minutes of scoring per training run and an overhead of roughly fifteen to twenty percent on a run of three to five hours. On that evidence the all-pairs setting was kept, with neighbours-only retained as a pre-planned fallback if a full run proves too costly.

## 3.5 The six measurements

The graph is summarised by six measurements, each a fraction in the interval [0,1]. Let *n* be the number of steps and *c* the conclusion step.

| Name | What it measures | Why it is in the reward |
|---|---|---|
| `support_density` | support links relative to *n*−1 | how much of the trace does inferential work |
| `attachment` | steps participating in any support link | detects floating assertions connected to nothing |
| `connectivity` | steps with a support path to *c* | does the reasoning actually lead to the answer |
| `support_depth` | longest support chain ending at *c*, capped at five | rewards chained inference over flat assertion lists |
| `conflict_rate` | contradiction links relative to *n*−1 | detects self-contradiction within a trace |
| `restatement_rate` | restatement links relative to *n*−1 | detects padding, the same point made twice |

The final two enter the reward inverted, since more of either is worse.

Two properties of this design are load-bearing. The first is that every measurement is continuous, which is a requirement rather than a preference. Group-relative policy optimisation [@shaoDeepSeekMathPushingLimits2024] computes each attempt's advantage relative to the mean of its group, so if all attempts in a group score identically, every advantage is zero and the model learns nothing from that question. Binary measurements make such ties frequent; fractional measurements make them rare. A binary reward would therefore fail silently, in a way that would be indistinguishable from the idea itself not working. The second is the degenerate guard: a trace with fewer than three steps, or with no identifiable conclusion, scores zero on all six measurements. Such a trace has no structure to assess, and scoring an absent graph would inject noise into the reward.

## 3.6 Conclusion detection: two corrections

Two of the six measurements, connectivity and support depth, depend on correctly identifying which step is the conclusion. Getting that identification right required two corrections, and the sequence is reported here in full because it is strong evidence of how the pipeline's difficulties were found and handled.

The original rule set the conclusion to the step matching the extracted answer. On the baseline corpus this went wrong (D-008): 93 percent of traces had the conclusion detected as the bare answer line, for example "The answer is No", and 39 percent had a connectivity of zero because the relation model never links reasoning to such a line. Two of the six measurements were therefore measuring a disconnected node. The fix was to drop trailing answer-announcement lines and take the conclusion to be the last remaining reasoning step. The answer continues to be extracted separately from the raw text, so correctness is unaffected.

The reasoning model exposed a second failure (D-011). Reasoning models state their verdict and then keep second-guessing, with trailing lines such as "But wait" and "Alternatively, suppose". Those trailing lines became the conclusion node in place of the verdict. On a hand-read trace, every real support link pointed into the verdict while the conclusion had been marked as a trailing fragment one step later, so connectivity measured paths to a node that nothing supported. The second fix extends the trailing-line drop to these self-doubt markers, applied only when they trail, since mid-trace reconsidering is genuine reasoning and is retained. The rule was verified to change no conclusion index on the instruction-tuned corpus, so the baseline is untouched and the two corpora remain comparable.

This sequence belongs in the report because the correlation study of Chapter 5 was re-run after the first correction and the null survived. A null result that survives the correction of a bug capable of producing it is a far stronger result than one reported before the check was made.

## 3.7 From measurements to reward

The six measurements are combined into a single reward as follows.

```
structural     = (weighted mean of the active measurements) x length_penalty
reward         = correctness + lambda x structural
length_penalty = min(1, length_target / trace_length)
```

Here correctness is one or zero, and lambda is 0.5. The two penalty measurements are inverted before the weighted mean, so that more conflict or more restatement lowers the score. An ablation setting can drop the two penalties entirely, leaving support only, which allows a later experiment to test whether it is the argument content that matters or merely the presence of any dense signal.

The weights come from the correlation study and from nowhere else. The scorer raises an error rather than run with unset weights. This guard is deliberate: hand-setting the weights and then reporting an improvement would be the single most damaging thing that could happen to the project's credibility, because it would make the reward a fitted quantity rather than a recovered one. In the same spirit, the length target is not a chosen constant but a percentile of the untrained model's own trace-length distribution, calibrated to the observed behaviour rather than imposed on it. The length penalty exists because all six measurements are fractions, so producing more text tends to create more links; without the penalty, verbosity would be a free way to raise the score.

One consequence of tying the weights to the correlation study must be recorded as a decision in its own right (logged as D-012). If the correlation study returns no predictive weight for any measurement, the reward as pre-registered is identically zero and cannot be trained. Resolving that is a decision to be taken with the supervisor, between uniform weights and weights informed by the per-family pattern, and the choice and its reasoning are documented rather than made silently in code.

## 3.8 Data

The data are organised into tiers at increasing distance from the training distribution, declared in `docs/splits.json`, and frozen and fingerprinted before any model saw them. The training tier is 80 percent of the items from the training families; the held-out tier is the remaining 20 percent of those same families; the transfer tier is entirely unseen families; and two external benchmarks, GSM8K [@cobbeTrainingVerifiersSolve2021] (250 items) and LogiQA [@liuLogiQAChallengeDataset2020], are added as evaluation-only tiers. Only the training tier is ever trained on.

Two revisions to this scheme were logged. The first (D-005) redefined the held-out tier: as originally written, held-out and transfer both measured unseen families and so tested the same thing twice. Making held-out a sample of the training families instead gives a clean progression, and turns the held-out to transfer gap into an interpretable quantity, the difference between having learned the questions and having learned to reason. The second (D-004, D-006) added GSM8K and LogiQA as external evaluation sets at the supervisor's suggestion. Breadth was added to evaluation rather than to training by deliberate choice: anything added to training is lost as a test, and GSM8K's arithmetic reasoning and LogiQA's argumentative reasoning pull in different directions, so together they bound where any effect holds.

Family selection followed a criterion stated before any reward was computed: a family is eligible only if its chains of thought contain natural-language inferential steps. On that basis a substantial fraction of the BIG-Bench Hard benchmark [@suzgunChallengingBIGBenchTasks2023] was excluded, in three groups each recorded with its reason: symbolic or mechanical tasks, where an argument graph over symbol manipulation is degenerate; tasks on which reasoning aloud is unlikely to help, such as simple lookups or pattern completions, whose inclusion would confound a reasoning intervention with tasks where a chain of thought adds nothing; and families judged weak or redundant. Excluding a large part of the benchmark is a significant scoping decision, and it is presented as one rather than left implicit.

## 3.9 Corpus generation and four data-quality corrections

The baseline corpus generates one trace per item, records the prompt, trace, extracted answer, correctness, extraction success, step splits, relations with confidences, all six measurements, and the decoding parameters, and counts extraction failures separately so that a non-answer is never scored as a wrong answer. In building it, four data-quality problems were found by manually inspecting real traces, and each was corrected before the corpus was trusted.

First, correctness used substring matching, so "valid" matched inside "invalid" and opposite answers were counted as correct, inflating correctness to a false 86 percent; this was fixed with whole-word matching. Second, sampling took the first *N* questions and so drew them all from a single family; this was fixed by shuffling with a fixed seed before sampling. Third, a regeneration appended to the existing file rather than replacing it, mixing old and new data; this was fixed with an explicit wipe and a fresh-generation option. Fourth, the base model answered in each task's own vocabulary, saying "plausible" rather than "yes", which was marked wrong and drove one family to 6 percent, below the 50 percent expected from random guessing on a binary task and so a signal of a labelling problem rather than genuine difficulty; this was fixed by stating the expected answer format per family and accepting synonyms.

The method note is stated plainly: correctness labels were validated by manual inspection of sampled traces at each stage, and four labelling or sampling issues were identified and corrected before use. The recovery is itself evidence that the fixes worked, with one family moving from 6 percent to 56 percent and another from 17 percent to 63 percent.

## 3.10 The change of base model

The plan specified an instruction-tuned model, Qwen2.5-3B-Instruct [@qwenQwen25TechnicalReport2025], prompted to reason step by step. The supervisor was firm that a project titled around reasoning models should use an actual reasoning model, on the grounds that an instruction-tuned model's chain of thought is prompt-induced and comparatively shallow, so structure measured on it is not representative of how a reasoning model reasons. The base model was therefore changed (D-009) to a native reasoning model [@qwenQwen3TechnicalReport2025], Qwen3-4B for the corpus and the smaller Qwen3-1.7B for the lighter training runs, which produce their chain of thought inside `<think>` tags.

What changed in the pipeline was confined to generation and extraction. The reasoning is mined from inside the tags and the answer read from after them; the decoding keeps the `<think>` markers rather than stripping them; sampling follows the Qwen3 model card for its thinking mode, at temperature 0.6, top-p 0.95, and top-k 20, with greedy decoding avoided as the card advises; and the generation budget was raised from 512 tokens to 2,048 and then to 4,096, because reasoning models think for much longer before answering and shorter budgets truncated them. What did not change was everything downstream of extraction: the relation model, the six measurements, the scorer, and both analyses re-run unmodified.

This change is treated as an asset rather than a disruption. The instruction-tuned corpus is retained as a baseline and the reasoning-model corpus becomes the comparison, which gives the dissertation a contrast it was not originally designed to have: whether argument structure reads differently on prompt-induced reasoning than on native reasoning. Both are reported.

The verbosity of the reasoning model also motivated a further design choice, recorded as a measured experiment rather than an assumption (D-010). Reasoning-model traces repeat themselves heavily, re-deriving the same chain and re-listing premises, which raises the question of whether the raw steps should be de-duplicated, or propositionalised into clean nodes, before relation identification. Rather than decide this by intuition, both arms are run, the direct one and the cleaned one, and the choice is made by measured parser fidelity: whichever recovers cleaner structure is kept, and the difference is reported.

## 3.11 Pre-registration and analysis plan

The hypotheses, the analyses, their order, and the rules for interpreting the results were all fixed in `docs/experiment_plan.md` before any data was seen, and every departure from that document is recorded in `docs/deviation_log.md` on the day it happened. This is what allows the central result to be reported as a finding rather than as the product of choices made after seeing the numbers.

The interpretation rule for the correlation study is quoted here, before any result is reported, so that the reader meets the decision rule before the number it governs:

| Composite AUC, length controlled | Reading | Action |
|---|---|---|
| > 0.60 | meaningful association | proceed as planned |
| 0.55 to 0.60 | weak association | proceed, temper H3 |
| < 0.55 | no predictive association | proceed, reframe the training experiment as testing whether optimisation *creates* structure the base model does not exploit |

Trace length is included as a covariate in the regression by deliberate design, because longer traces have more steps and therefore more links, so any raw association between structure and correctness could be verbosity rather than structure. This is the first objection an examiner is likely to raise, and it is pre-empted by the design rather than answered defensively after the fact. Effect sizes with confidence intervals are reported first and p-values as supporting evidence, never as a verdict; and where the unit of analysis is the training seed, the resulting limitation on statistical power is stated openly rather than obscured.
