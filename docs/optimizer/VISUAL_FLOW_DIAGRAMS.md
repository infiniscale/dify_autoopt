# Visual Flow Diagram: Multi-Strategy Iterative Optimization

**Comprehensive Flow Visualization**

---

## 1. Current Implementation (Broken)

```
┌─────────────────────────────────────────────────────────────┐
│           run_optimization_cycle() - CURRENT                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: workflow_id, strategy="clarity_focus"               │
│                                                             │
│  ┌─────────────────────────────────────────┐               │
│  │ config = OptimizationConfig(            │               │
│  │   strategies=[CLARITY, EFFICIENCY],     │ ❌ IGNORED    │
│  │   max_iterations=3,                     │ ❌ IGNORED    │
│  │   min_confidence=0.7,                   │ ❌ IGNORED    │
│  │   score_threshold=80.0                  │ ✅ USED       │
│  │ )                                       │               │
│  └─────────────────────────────────────────┘               │
│                                                             │
│  prompts = extract_prompts(workflow_id)                     │
│                                                             │
│  for prompt in prompts:                                     │
│      analysis = analyze(prompt)                             │
│                                                             │
│      if analysis.score < 80.0:  # Uses score_threshold     │
│          result = engine.optimize(                          │
│              prompt,                                        │
│              strategy="clarity_focus"  # Fixed, not from config│
│          )                                                  │
│          # ❌ Runs ONCE (max_iterations ignored)           │
│          # ❌ No confidence check (min_confidence ignored)  │
│          # ❌ No strategy list trial (strategies ignored)   │
│                                                             │
│          create_version(result)                             │
│          patches.append(create_patch(result))               │
│                                                             │
│  return patches                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Problems**:
- ❌ `config.strategies` list never used
- ❌ `config.max_iterations` never checked
- ❌ `config.min_confidence` never validated

---

## 2. New Implementation (Fixed)

```
┌─────────────────────────────────────────────────────────────────────────┐
│              run_optimization_cycle() - NEW DESIGN                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: workflow_id, strategy?, config?                                │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ STEP 1: Resolve Configuration                                 │     │
│  ├───────────────────────────────────────────────────────────────┤     │
│  │                                                                │     │
│  │ if strategy is not None:                                      │     │
│  │     # Backward compatibility mode                             │     │
│  │     effective_config = OptimizationConfig(                    │     │
│  │         strategies=[strategy],    # Single strategy           │     │
│  │         max_iterations=1,         # Single run                │     │
│  │         min_confidence=0.0        # No filtering              │     │
│  │     )                                                          │     │
│  │     LOG: "Using legacy single-strategy mode"                  │     │
│  │                                                                │     │
│  │ elif config is not None:                                      │     │
│  │     # New multi-strategy mode                                 │     │
│  │     effective_config = config                                 │     │
│  │     LOG: "Using multi-strategy mode: {N} strategies"          │     │
│  │                                                                │     │
│  │ else:                                                          │     │
│  │     # Default mode                                            │     │
│  │     effective_config = OptimizationConfig()  # AUTO strategy  │     │
│  │     LOG: "Using default configuration"                        │     │
│  │                                                                │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ STEP 2: Extract Prompts                                       │     │
│  ├───────────────────────────────────────────────────────────────┤     │
│  │                                                                │     │
│  │ prompts = extract_prompts(workflow_id)                        │     │
│  │ LOG: "Extracted {N} prompts"                                  │     │
│  │                                                                │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ STEP 3: Optimize Each Prompt                                  │     │
│  ├───────────────────────────────────────────────────────────────┤     │
│  │                                                                │     │
│  │ patches = []                                                   │     │
│  │                                                                │     │
│  │ for prompt in prompts:                                         │     │
│  │                                                                │     │
│  │     # 3.1: Analyze baseline                                   │     │
│  │     baseline_analysis = analyze(prompt)                        │     │
│  │     create_baseline_version(prompt, baseline_analysis)         │     │
│  │                                                                │     │
│  │     # 3.2: Check if optimization needed                       │     │
│  │     if baseline_analysis.score >= effective_config.score_threshold:│
│  │         LOG: "Prompt already good enough, skipping"           │     │
│  │         continue                                              │     │
│  │                                                                │     │
│  │     # 3.3: Try all strategies                                 │     │
│  │     best_result = None                                        │     │
│  │                                                                │     │
│  │     ┌──────────────────────────────────────────────────────┐  │     │
│  │     │ FOR EACH STRATEGY IN effective_config.strategies    │  │     │
│  │     ├──────────────────────────────────────────────────────┤  │     │
│  │     │                                                      │  │     │
│  │     │ LOG: "Trying strategy '{strategy}'"                 │  │     │
│  │     │                                                      │  │     │
│  │     │ ┌────────────────────────────────────────────────┐  │  │     │
│  │     │ │ result = _optimize_with_iterations(           │  │  │     │
│  │     │ │     prompt=prompt,                             │  │  │     │
│  │     │ │     strategy=strategy,                         │  │  │     │
│  │     │ │     max_iterations=effective_config.max_iterations│ │  │     │
│  │     │ │     min_confidence=effective_config.min_confidence│ │  │     │
│  │     │ │ )                                              │  │  │     │
│  │     │ │                                                │  │  │     │
│  │     │ │ ITERATION LOOP:                                │  │  │     │
│  │     │ │ ┌────────────────────────────────────────┐     │  │  │     │
│  │     │ │ │ current_prompt = prompt                │     │  │  │     │
│  │     │ │ │ best = None                            │     │  │  │     │
│  │     │ │ │                                        │     │  │  │     │
│  │     │ │ │ for i in range(max_iterations):        │     │  │  │     │
│  │     │ │ │     LOG: "Iteration {i+1}/{max}"       │     │  │  │     │
│  │     │ │ │                                        │     │  │  │     │
│  │     │ │ │     result = engine.optimize(          │     │  │  │     │
│  │     │ │ │         current_prompt, strategy       │     │  │  │     │
│  │     │ │ │     )                                  │     │  │  │     │
│  │     │ │ │                                        │     │  │  │     │
│  │     │ │ │     # Track best                       │     │  │  │     │
│  │     │ │ │     if result.improvement > best:      │     │  │  │     │
│  │     │ │ │         best = result                  │     │  │  │     │
│  │     │ │ │                                        │     │  │  │     │
│  │     │ │ │     # Check success condition          │     │  │  │     │
│  │     │ │ │     if result.confidence >= min_confidence:│  │  │     │
│  │     │ │ │         LOG: "Confidence met!"         │     │  │  │     │
│  │     │ │ │         return result  ✅ SUCCESS      │     │  │  │     │
│  │     │ │ │                                        │     │  │  │     │
│  │     │ │ │     # Check convergence                │     │  │  │     │
│  │     │ │ │     if i > 0 and result.improvement <= 0:│   │  │  │     │
│  │     │ │ │         LOG: "No improvement, stop"    │     │  │  │     │
│  │     │ │ │         break  📉 CONVERGED            │     │  │  │     │
│  │     │ │ │                                        │     │  │  │     │
│  │     │ │ │     # Prepare next iteration           │     │  │  │     │
│  │     │ │ │     current_prompt = create_prompt(    │     │  │  │     │
│  │     │ │ │         text=result.optimized_prompt   │     │  │  │     │
│  │     │ │ │     )                                  │     │  │  │     │
│  │     │ │ │                                        │     │  │  │     │
│  │     │ │ │ return best  ⏱️ MAX ITERATIONS REACHED│     │  │  │     │
│  │     │ │ └────────────────────────────────────────┘     │  │  │     │
│  │     │ └────────────────────────────────────────────────┘  │  │     │
│  │     │                                                      │  │     │
│  │     │ # Compare with current best                         │  │     │
│  │     │ if _is_better_result(result, best_result):          │  │     │
│  │     │     best_result = result                            │  │     │
│  │     │     LOG: "New best: strategy={s}, conf={c}"         │  │     │
│  │     │                                                      │  │     │
│  │     └──────────────────────────────────────────────────────┘  │     │
│  │                                                                │     │
│  │     # 3.4: Accept result if meets confidence threshold        │     │
│  │     if best_result and best_result.confidence >= min_confidence:│   │
│  │         create_optimized_version(best_result)                  │     │
│  │         patch = create_patch(best_result)                      │     │
│  │         patches.append(patch)                                  │     │
│  │         LOG: "Accepted optimization: conf={c}, imp={i}"        │     │
│  │     else:                                                      │     │
│  │         LOG: "No acceptable optimization found"                │     │
│  │                                                                │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  return patches                                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Helper Method: _optimize_with_iterations()

```
┌─────────────────────────────────────────────────────────────┐
│        _optimize_with_iterations(prompt, strategy,          │
│                 max_iterations, min_confidence)             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input:                                                     │
│    - prompt: Original prompt to optimize                   │
│    - strategy: Single strategy name (e.g., "clarity_focus")│
│    - max_iterations: Maximum attempts (e.g., 3)            │
│    - min_confidence: Success threshold (e.g., 0.7)         │
│                                                             │
│  Initialize:                                                │
│    current_prompt = prompt                                  │
│    best_result = None                                       │
│    best_score = -infinity                                   │
│                                                             │
│  ┌───────────────────────────────────────────────────┐     │
│  │ FOR iteration IN range(max_iterations)            │     │
│  ├───────────────────────────────────────────────────┤     │
│  │                                                   │     │
│  │  LOG: "Iteration {i+1}/{max_iterations}"          │     │
│  │                                                   │     │
│  │  # Optimize current prompt                        │     │
│  │  result = engine.optimize(current_prompt, strategy)│    │
│  │                                                   │     │
│  │  # Track best result                              │     │
│  │  if result.improvement_score > best_score:        │     │
│  │      best_score = result.improvement_score        │     │
│  │      best_result = result                         │     │
│  │                                                   │     │
│  │  # SUCCESS: Confidence threshold met              │     │
│  │  if result.confidence >= min_confidence:          │     │
│  │      LOG: "✅ Confidence met: {conf} >= {min}"    │     │
│  │      return result  # Early exit                  │     │
│  │                                                   │     │
│  │  # CONVERGENCE: No improvement                    │     │
│  │  if iteration > 0 and result.improvement_score <= 0:│   │
│  │      LOG: "📉 No improvement, stopping early"     │     │
│  │      break                                        │     │
│  │                                                   │     │
│  │  # Prepare next iteration                         │     │
│  │  current_prompt = Prompt(                         │     │
│  │      text=result.optimized_prompt,                │     │
│  │      ... # Copy other fields                      │     │
│  │  )                                                │     │
│  │                                                   │     │
│  └───────────────────────────────────────────────────┘     │
│                                                             │
│  # MAX ITERATIONS REACHED                                  │
│  if best_result:                                            │
│      LOG: "⏱️ Max iterations reached. Best conf={c}"       │
│                                                             │
│  return best_result  # May be None if all failed           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Helper Method: _is_better_result()

```
┌─────────────────────────────────────────────────────────────┐
│   _is_better_result(candidate, current_best, min_confidence)│
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input:                                                     │
│    - candidate: New optimization result                     │
│    - current_best: Current best result (or None)            │
│    - min_confidence: Minimum confidence threshold           │
│                                                             │
│  Returns: True if candidate is better                       │
│                                                             │
│  ┌───────────────────────────────────────────────────┐     │
│  │ STEP 1: Handle None case                          │     │
│  ├───────────────────────────────────────────────────┤     │
│  │                                                   │     │
│  │ if current_best is None:                          │     │
│  │     return True  # First result is always best    │     │
│  │                                                   │     │
│  └───────────────────────────────────────────────────┘     │
│                                                             │
│  ┌───────────────────────────────────────────────────┐     │
│  │ STEP 2: Priority 1 - Confidence Threshold         │     │
│  ├───────────────────────────────────────────────────┤     │
│  │                                                   │     │
│  │ candidate_meets = (candidate.conf >= min_conf)    │     │
│  │ current_meets = (current_best.conf >= min_conf)   │     │
│  │                                                   │     │
│  │ if candidate_meets and not current_meets:         │     │
│  │     return True  # Candidate passes, current fails│     │
│  │                                                   │     │
│  │ if current_meets and not candidate_meets:         │     │
│  │     return False  # Current passes, candidate fails│    │
│  │                                                   │     │
│  └───────────────────────────────────────────────────┘     │
│                                                             │
│  ┌───────────────────────────────────────────────────┐     │
│  │ STEP 3: Priority 2 - Overall Score                │     │
│  ├───────────────────────────────────────────────────┤     │
│  │                                                   │     │
│  │ candidate_score = candidate.metadata["optimized_score"]│ │
│  │ current_score = current_best.metadata["optimized_score"]│ │
│  │                                                   │     │
│  │ if candidate_score > current_score + 1.0:         │     │
│  │     return True  # Significantly better score     │     │
│  │                                                   │     │
│  │ if current_score > candidate_score + 1.0:         │     │
│  │     return False  # Significantly worse score     │     │
│  │                                                   │     │
│  └───────────────────────────────────────────────────┘     │
│                                                             │
│  ┌───────────────────────────────────────────────────┐     │
│  │ STEP 4: Priority 3 - Confidence (Tie-breaker)     │     │
│  ├───────────────────────────────────────────────────┤     │
│  │                                                   │     │
│  │ return candidate.confidence > current_best.confidence│   │
│  │                                                   │     │
│  └───────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Execution Flow Examples

### Example 1: Legacy Single-Strategy Mode

```
User Call:
  run_optimization_cycle("wf_001", strategy="clarity_focus")

Resolution:
  effective_config = OptimizationConfig(
      strategies=[CLARITY_FOCUS],
      max_iterations=1,
      min_confidence=0.0
  )

Execution:
  Prompt 1:
    Strategy: clarity_focus
      Iteration 1: optimize() → result (conf=0.65, imp=8.5)
      ✅ Accept (min_confidence=0.0)
    Best: clarity_focus (conf=0.65)

  Prompt 2:
    Strategy: clarity_focus
      Iteration 1: optimize() → result (conf=0.72, imp=10.2)
      ✅ Accept (min_confidence=0.0)
    Best: clarity_focus (conf=0.72)

Result: 2 patches generated
```

### Example 2: Multi-Strategy with Iterations

```
User Call:
  config = OptimizationConfig(
      strategies=[CLARITY_FOCUS, EFFICIENCY_FOCUS],
      max_iterations=3,
      min_confidence=0.7
  )
  run_optimization_cycle("wf_001", config=config)

Resolution:
  effective_config = config (as provided)

Execution:
  Prompt 1:
    Strategy 1: clarity_focus
      Iteration 1: optimize() → result (conf=0.60, imp=8.0)
        ❌ Confidence not met (0.60 < 0.7)
      Iteration 2: optimize() → result (conf=0.75, imp=10.5)
        ✅ Confidence met (0.75 >= 0.7) - STOP iterating
      Best for clarity_focus: (conf=0.75, imp=10.5)

    Strategy 2: efficiency_focus
      Iteration 1: optimize() → result (conf=0.55, imp=6.0)
        ❌ Confidence not met (0.55 < 0.7)
      Iteration 2: optimize() → result (conf=0.62, imp=6.8)
        ❌ Confidence not met (0.62 < 0.7)
      Iteration 3: optimize() → result (conf=0.68, imp=7.2)
        ❌ Confidence not met (0.68 < 0.7)
      Best for efficiency_focus: (conf=0.68, imp=7.2)

    Compare:
      clarity_focus: conf=0.75 ✅ meets threshold
      efficiency_focus: conf=0.68 ❌ fails threshold

    Select: clarity_focus (only one meeting threshold)
    ✅ Accept and create patch

  Prompt 2:
    Strategy 1: clarity_focus
      Iteration 1: optimize() → result (conf=0.50, imp=5.0)
        ❌ Confidence not met
      Iteration 2: optimize() → result (conf=0.52, imp=5.2)
        ❌ No improvement (5.2 - 5.0 = 0.2 ≈ 0)
        📉 STOP early (convergence)
      Best for clarity_focus: (conf=0.52, imp=5.2)

    Strategy 2: efficiency_focus
      Iteration 1: optimize() → result (conf=0.45, imp=4.0)
        ❌ Confidence not met
      Iteration 2: optimize() → result (conf=0.48, imp=4.3)
        ❌ Confidence not met
      Iteration 3: optimize() → result (conf=0.50, imp=4.5)
        ❌ Confidence not met
      Best for efficiency_focus: (conf=0.50, imp=4.5)

    Compare:
      clarity_focus: conf=0.52 ❌ fails threshold
      efficiency_focus: conf=0.50 ❌ fails threshold

    Select: None (both fail threshold)
    ⚠️ Skip this prompt (log warning)

Result: 1 patch generated
```

---

## 6. Version History Example

```
Prompt: wf_001_llm_1

Version History:
  v1.0.0 - Baseline
    author: baseline
    score: 68.0
    optimization_result: None

  v1.1.0 - First optimization (clarity_focus, iteration 1)
    author: optimizer
    strategy: clarity_focus
    iteration: 1
    is_intermediate: True
    score: 76.0
    confidence: 0.60
    parent: 1.0.0

  v1.2.0 - Second optimization (clarity_focus, iteration 2)
    author: optimizer
    strategy: clarity_focus
    iteration: 2
    is_intermediate: False  # Final accepted result
    score: 78.5
    confidence: 0.75
    parent: 1.1.0

Note: efficiency_focus iterations not saved because clarity_focus was selected as best
```

---

## 7. Performance Flow

```
Worst-Case Scenario:
  - 10 prompts
  - 3 strategies
  - 3 iterations each
  - All iterations run (no early stop)

Total Optimizations: 10 × 3 × 3 = 90 calls

Time Estimate: 90 × 200ms = 18 seconds

Optimized Scenario (with early stop):
  - 10 prompts
  - 3 strategies
  - Average 1.5 iterations (early stop at 50% rate)
  - Some strategies skipped (high confidence found)

Total Optimizations: 10 × 2 × 1.5 = 30 calls

Time Estimate: 30 × 200ms = 6 seconds (67% faster)
```

---

## 8. Decision Tree

```
┌─────────────────────────────────────────────────────┐
│ run_optimization_cycle(workflow_id, strategy?, config?)│
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │ strategy parameter provided? │
        └──────┬───────────────┬────────┘
               │ YES           │ NO
               ▼               ▼
        ┌─────────────┐  ┌──────────────┐
        │ LEGACY MODE │  │ config provided? │
        │             │  └─────┬──────┬─────┘
        │ Single-strategy │    │ YES  │ NO
        │ 1 iteration     │    ▼      ▼
        │ No confidence   │ ┌────┐ ┌──────┐
        │ check           │ │NEW │ │DEFAULT│
        └─────────────────┘ │MODE│ │ MODE │
                            └────┘ └──────┘
                               │       │
                               ▼       ▼
                        Multi-strategy  AUTO
                        N iterations   1 iteration
                        Confidence check  No check

All paths converge:
    ▼
┌───────────────────────┐
│ Extract prompts       │
├───────────────────────┤
│ For each prompt:      │
│   Baseline analysis   │
│   If score < threshold:│
│     Try strategies    │
│     Select best       │
│     Check confidence  │
│     Accept or reject  │
└───────────────────────┘
    ▼
Return patches
```

---

## 9. State Diagram

```
┌─────────┐
│  START  │
└────┬────┘
     │
     ▼
┌────────────────┐
│ Extract Prompts│
└────┬───────────┘
     │
     ▼
┌──────────────────────┐
│ For Each Prompt:     │
│                      │
│ ┌────────────────┐   │
│ │ Analyze        │   │
│ │ Baseline       │   │
│ └───┬────────────┘   │
│     │                │
│     ▼                │
│ ┌────────────────┐   │
│ │ Score >=       │   │
│ │ Threshold?     │   │
│ └─┬────────┬─────┘   │
│   │ YES    │ NO      │
│   ▼        ▼         │
│ ┌─────┐ ┌──────────┐ │
│ │ SKIP│ │ Optimize │ │
│ └─────┘ │ Multi-   │ │
│         │ Strategy │ │
│         └────┬─────┘ │
│              │       │
│              ▼       │
│         ┌──────────┐ │
│         │ Best     │ │
│         │ Result   │ │
│         │ Found?   │ │
│         └─┬────┬───┘ │
│           │YES │NO   │
│           ▼    ▼     │
│         ┌───┐┌────┐  │
│         │Conf│SKIP│  │
│         │>=  │    │  │
│         │Min?│    │  │
│         └┬──┬┘    │  │
│          │Y │N    │  │
│          ▼  ▼     │  │
│       ┌───┐┌───┐  │  │
│       │ACC│SKIP│  │  │
│       │EPT│    │  │  │
│       └───┘└───┘  │  │
└──────────────────────┘
     │
     ▼
┌────────────┐
│Return Patches│
└────┬───────┘
     │
     ▼
┌─────┐
│ END │
└─────┘
```

---

**Document**: VISUAL_FLOW_DIAGRAMS.md
**Companion Documents**:
- ARCHITECTURE_DESIGN_MULTI_STRATEGY_ITERATION.md (full design)
- ARCHITECTURE_DESIGN_SUMMARY.md (quick reference)

**Date**: 2025-11-18
**Author**: Senior System Architect
