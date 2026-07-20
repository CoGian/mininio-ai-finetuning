# AI Integration Plan — Mininio Carb & Insulin Calc

## Overview

Add a self-hosted, on-device LLM that lets users describe their meal in natural language and have the app automatically run the full calculation pipeline. The LLM acts as a smart input parser + action sequencer, calling the same `DietViewModel` methods a human would. **All math still goes through `DietCalculator.kt`** — the LLM never does arithmetic.

---

## Key Decisions

| Decision | Choice |
|----------|--------|
| Model | Two candidates evaluated in parallel — winner selected post fine-tuning |
| Candidate A (Primary) | LFM2.5-1.2B-Instruct (Liquid AI) |
| Candidate B (Fallback) | Gemma 4 E2B QAT (Google) |
| Model delivery | Play Asset Delivery (install-time) |
| UI placement | New 5th tab: "AI Assist" |
| Languages | All 10 from start (en, el, fr, es, hi, it, pt, zh, de, ja) |
| Meal time default | Infer from device time; respect explicit hour/meal mention |
| Training data | LLM-assisted synthetic generation (Gemini API) |

---

## 1. Model Candidates — Dual Evaluation Path

We will fine-tune and evaluate two models in parallel using the same 8,000-conversation training dataset, then select the winner based on a weighted score across accuracy, latency, memory, and license terms.

### Candidate A: LFM2.5-1.2B-Instruct **(Primary)**

| Aspect | Detail |
|--------|--------|
| Parameters | 1.2B |
| Disk size | ~719MB (Q4_0 GGUF) |
| Peak RAM | ~719MB |
| Tool calling (pre-FT) | BFCLv3 49.12 — native `<\|tool_call_start\|>` / `<\|tool_call_end\|>` tokens |
| Instruction following | IFEval 86.23 — strong at following complex prompts |
| Conversation quality | Good — general-purpose instruct-tuned model |
| Languages (pre-FT) | 8 of 10: en, ar, zh, fr, de, ja, ko, es |
| Languages to teach via FT | 4: Greek (el), Hindi (hi), Italian (it), Portuguese (pt) |
| Android runtime | llama.cpp (GGUF via JNI) or LEAP SDK |
| NPU support | Qualcomm (NexaML), AMD |
| License | $10M annual revenue cap (free below) |
| S25U CPU speed (Q4_0) | 70 tok/s decode, 335 tok/s prefill |
| **FT effort** | **Low** — model already knows tool calling + instruction following. Fine-tune for: 4 missing languages, carb-counting domain terminology, our 6 tool schemas |
| **FT framework** | Unsloth LoRA SFT (5+ published Colab recipes) |
| **FT time** | ~4-6 hours on A100 40GB |

**Why it's the primary candidate**: LFM2.5 already knows **how** to call tools (BFCLv3 49.12 pre-FT) and **how** to follow instructions (IFEval 86.23). We only need to teach it **which** tools and **what** the carb domain looks like. Its strongest advantage is natural conversation ability — it can ask clarifying questions ("Did you mean rice cooked or raw?"), handle corrections gracefully, and present results conversationally. The 4 missing languages are reactivated from latent pre-training knowledge via fine-tuning.

### Candidate B: Gemma 4 E2B QAT **(Fallback)**

| Aspect | Detail |
|--------|--------|
| Effective parameters | ~2B (MoE architecture) |
| Disk size | <1GB (mobile QAT format) |
| Peak RAM | <1GB |
| Tool calling (pre-FT) | None — general LLM, no tool-calling specialization |
| Languages (pre-FT) | Broad multilingual (all 10 likely covered) |
| Android runtime | LiteRT-LM (Google first-party SDK) |
| NPU support | Google Tensor, Qualcomm |
| License | Gemma — permissive, no revenue cap |
| Speed | TBD (mobile QAT format benchmarks pending) |
| **FT effort** | **Medium** — model needs to learn: tool calling format from scratch, carb domain terminology, our 6 tool schemas. Same 8,000-conversation dataset. |
| **FT framework** | HuggingFace Transformers QLoRA / Unsloth |
| **FT time** | ~6-8 hours on A100 40GB |

**Why it's the fallback**: Gemma 4 E2B has two advantages over LFM2.5 — no revenue cap on the license, and first-class Android integration via Google's LiteRT-LM. However it starts from zero on tool calling, needing to learn the format, the sequencing, and the domain all at once. The larger parameter count (2B effective) may help compensate for this gap.

### Side-by-Side Comparison

| | A: LFM2.5-1.2B-Instruct | B: Gemma 4 E2B QAT |
|---|---|---|
| Tool calling baseline | Exists (BFCLv3 49.12) | None — full training needed |
| Conversation baseline | Good (IFEval 86.23) | Good (general LLM) |
| Language gaps | 4 of 10 missing | All 10 likely covered |
| FT data needed | 8,000 convos (domain + 4 langs) | 8,000 convos (tool format + domain) |
| FT risk | Lower | Higher |
| Android deps | llama.cpp or LEAP | LiteRT-LM |
| License | $10M annual cap | No cap |
| Disk size | ~719MB | <1GB |
| Decision criteria | Better at dialogue + tool calls | Better licensing + Google ecosystem |

---

## 2. Tools (Functions)

Six tools total. `calculate_final` absorbs `set_meal_time` and `set_blood_glucose` as optional parameters. Batch versions (`search_foods`, `add_foods_to_tally`, `remove_foods_from_tally`) replace singular versions to reduce round-trips.

### 2.1 `search_foods`

Searches the nutrition database for multiple food names in one call.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| queries | string[] | Yes | Food names to search (e.g. `["potatoes", "bread"]`) |

**Returns**: Array of results per query — each result `{id, name, carbs_per_100g, carbs_per_piece, has_grams_mode, has_pieces_mode}`. If a query has no match, that slot is an empty array `[]`.

---

### 2.2 `add_foods_to_tally`

Adds multiple food items to the tally at once. Each item's carbs are calculated via `(quantity × food_carbs) / standard_quantity`.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| items | array | Yes | List of `{food_id: int, quantity: float, unit: "g"\|"ml"\|"pcs"\|"cup"\|"tbsp"\|"slice"}` |

**Precondition**: Every `food_id` must be in the `KNOWN FOOD IDS` set (populated by prior `search_foods` calls). The harness rejects any unknown ID with an error.

**Returns**: `{entries: [{entry_id, food_name, quantity, unit, carbs}], tally_total: float}`

---

### 2.3 `remove_foods_from_tally`

Removes multiple entries from the tally by their entry IDs.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| entry_ids | int[] | Yes | Entry IDs to remove (from `CURRENT TALLY` context or `add_foods_to_tally` returns) |

**Precondition**: Every `entry_id` must exist in the current tally. The harness rejects invalid IDs with an error.

**Returns**: `{removed: int, tally_total: float}`

---

### 2.4 `calculate_final`

Computes the final insulin dose. Absorbs meal time and blood glucose setting — no need for separate setter calls.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| meal_time | string | No | "morning", "midday", or "evening". Omitted/null → inferred from device time. |
| meal_hour | int | No | Specific hour mentioned by user (0-23). Null if not specified. |
| blood_glucose | float | No | Blood glucose in mg/dL. Omitted/null → no glucose correction. |

**Meal time handling logic:**

| User says | LLM passes | Result |
|-----------|-----------|--------|
| *(nothing about meal time)* | `meal_time=null` | Tool infers from device time |
| "lunch at 2pm" | `meal_time="midday", meal_hour=14` | midday, divider 15 |
| "breakfast" / "morning meal" | `meal_time="morning"` | morning, divider 14 |
| "dinner at 8pm" | `meal_time="evening", meal_hour=20` | evening, divider 12 |

**Precondition**: Tally must not be empty. The harness returns an error if no foods have been added.

**Returns**: `{final_result, food_insulin, glucose_correction, glucose_skipped, tally_total, meal_divider, meal_time, meal_hour, blood_glucose, threshold, baseline, divisor, breakdown_food, breakdown_glucose}`

---

### 2.5 `get_tally_summary`

Gets current calculation state. Rarely needed in normal flow (state is injected into user messages via context block) — kept as a safety net for recovery after complex sequences.

**No parameters.**  
**Returns**: `{entries: [{entry_id, food_name, quantity, unit, carbs}], total_carbs, food_insulin, meal_time, meal_hour, blood_glucose, glucose_enabled}`

---

### 2.6 `clear_all`

Clears all calculation data and starts fresh. Also clears the session's `knownFoodIds` set.

**No parameters.**  
**Returns**: `{success: bool}`

---

### 2.7 Dependency Rules (Harness Enforcement)

The harness validates preconditions before executing any tool:

```
search_foods ──→ populates KNOWN FOOD IDS ──→ add_foods_to_tally
                                                  │
                                                  ├──→ populates CURRENT TALLY entries
                                                  │
                                                  ▼
                                         calculate_final
                                       (requires: tally not empty)

remove_foods_from_tally ← requires entry_ids exist in CURRENT TALLY
get_tally_summary ← safety net, returns same data already in context
clear_all ← no deps, clears KNOWN FOOD IDS + tally
```

| Tool | Harness validates before execution |
|------|----------------------------------|
| `search_foods` | *(none)* — always OK |
| `add_foods_to_tally` | Every `food_id` must be in session `knownFoodIds` (populated by prior search results). Rejected IDs return error. |
| `remove_foods_from_tally` | Every `entry_id` must exist in `DietUiState.tallyEntries`. Invalid IDs return error. |
| `calculate_final` | `tallyEntries.isNotEmpty()`. If empty → error: "Add at least one food first." |
| `get_tally_summary` | *(none)* |
| `clear_all` | *(none)* |

**`knownFoodIds`**: Session-scoped `MutableSet<Long>` in `ToolExecutor`. Populated by every `search_foods` response. Cleared on `clear_all()`. Prevents LLM from hallucinating food IDs — even a guessed ID is rejected unless it came from a prior search.

---

### 2.8 Context Injection — Tally + Known Foods in User Messages

The harness injects the current tally state and known food IDs into **every new user message** before sending to the LLM. Tool responses get no injection. The LLM always knows the current state without calling `get_tally_summary`.

**Injected context block format** (appended to user message):

```
[CURRENT TALLY: 2 items, 47.0g total]
  Potatoes (boiled) 100g = 17.0g (entry_id: 1)
  Bread (white) 2pcs = 30.0g (entry_id: 2)

[KNOWN FOOD IDS: Potatoes(12), Bread(25)]
```

When empty:
```
[CURRENT TALLY: empty]
[KNOWN FOOD IDS: none]
```

**When it happens**: Only at the user-message boundary — once per turn, after the user sends a message and before the agentic loop starts. Not injected into tool responses within the loop.

---

### 2.9 System Prompt — Settings + Context Rules

The user's current settings are injected into the system prompt. The `AgenticHarness` rebuilds it reactively whenever settings change.

```
You are a carb counting assistant for people with diabetes, integrated into the Mininio app.
Help users calculate insulin doses by searching the nutrition database, adding foods to a
tally, and computing glucose corrections.

CURRENT USER SETTINGS:
- Glucose: threshold=130.0 mg/dL, baseline=100.0 mg/dL, divisor=40.0 mg/dL per unit
- Meal dividers: Morning=14, Midday=15, Evening=12
- Meal time ranges: Morning (4:00-12:00), Midday (12:00-17:00), Evening (17:00-4:00)

CALCULATION RULES:
- Carbs per food = (quantity × food_carbs) / standard_quantity
- Food insulin = tally_total / meal_divider
- Glucose correction = max(0, (blood_glucose - baseline) / divisor) only if bg ≥ threshold
- Final dose = food_insulin + glucose_correction

IMPORTANT: Every user message includes a [CURRENT TALLY] and [KNOWN FOOD IDS] block. This
is the AUTHORITATIVE source of truth — trust it over your own memory. You do not need to
call get_tally_summary to discover state unless you've lost track.

Always use the provided functions. If a food name matches multiple results, ask the user to
clarify. If quantity is missing, ask. Call calculate_final with meal_time and blood_glucose
as parameters — no separate setter calls are needed.
```

The harness replaces placeholder values with actuals from `SettingsRepository`.

---

## 3. Agentic Loop (ReAct Pattern + Context Injection)

### 3.1 Harness Pseudocode

```
runAgent(userMessage):
  // --- Inject context at user-message boundary ONLY ---
  contextBlock = buildContextBlock(dietViewModel.state, knownFoodIds)
  enrichedMessage = userMessage + "\n\n" + contextBlock
  messages.add(user: enrichedMessage)

  maxIterations = 15

  while maxIterations-- > 0:
    response = llm.generate(messages, toolSchemas)

    if response is TEXT:
      return response.text  // LLM is done

    toolCalls = response.toolCalls
    groups = partitionByDependency(toolCalls)

    for group in groups:
      results = coroutineScope {
        group.map { call ->
          async {
            validate(call)         // check preconditions (knownFoodIds, tally state)
            execute(call)          // run via ToolExecutor → updates DietUiState
          }
        }.awaitAll()
      }
      // Update session state from results
      for (i in 0..results.lastIndex):
        if group[i].name == "search_foods":
          knownFoodIds.addAll(results[i].ids)
        if group[i].name == "clear_all":
          knownFoodIds.clear()
      // Append tool results to message history (NO context injection here)
      for result in results:
        messages.add(tool_response: result)
      emit(AgentStep.ToolResults(results))

  return fallbackMessage
```

**Key**: `buildContextBlock()` is called **once per user message**, at the start. Tool responses within the loop get no context injection — the LLM tracks state changes from tool return values.

### 3.2 Parallel Calling Rules (Updated for 6 Tools)

| Tool A | Tool B | Can parallel? | Rationale |
|--------|--------|:---:|-----------|
| `search_foods` | `search_foods` | **Yes** | Already batch — rarely duplicated, but safe if it happens |
| `add_foods_to_tally` | `add_foods_to_tally` | **Yes** | Already batch — IDs confirmed from `KNOWN FOOD IDS` |
| `calculate_final` | anything | **No** | Must be last — depends on all accumulated state |
| `clear_all` | anything | **No** | Destructive — must serialize entirely |
| `remove_foods_from_tally` | `add_foods_to_tally` | **No** | Could conflict on the same tally — serialize |
| `get_tally_summary` | `remove_foods_from_tally` | **No** | `remove` depends on knowing current `entry_id`s first |

**With batch tools, parallelism is rarely needed** — a typical flow is 4 iterations total (see Section 4). Parallel execution still works as a safety net if the LLM issues multiple calls in one turn.

**State machine**: `IDLE → THINKING → TOOL_CALLS → TOOL_RESULTS → (loop) → REPLYING → DONE`

**Critical invariants**:
- `ToolExecutor` is a thin adapter — it calls existing `DietViewModel` methods
- The same `DietUiState` drives both manual and AI flows
- `knownFoodIds` (session-scoped `MutableSet<Long>`) prevents hallucinated food IDs
- `buildContextBlock()` injects fresh state at every user message boundary

---

## 4. Agentic Loop — Full Examples

### 4.1 Standard Meal with Glucose

```
User: "I ate 100g of potatoes and 2 slices of bread. My blood sugar is 140."

Harness injects context → LLM receives:
  "I ate 100g of potatoes and 2 slices of bread. My blood sugar is 140.

   [CURRENT TALLY: empty]
   [KNOWN FOOD IDS: none]"

[1] LLM → search_foods(["potatoes", "bread"])
    Tool → [{id:12, name:"Potatoes (boiled)", carbs_per_100g:17.0},
            {id:25, name:"Bread (white)", carbs_per_piece:15.0}]
    → knownFoodIds updated to {12, 25}

[2] LLM → add_foods_to_tally([{12, 100, "g"}, {25, 2, "slice"}])
    → Harness validates: 12 and 25 are in knownFoodIds → OK
    Tool → {entries: [{1,"Potatoes",100,"g",17.0}, {2,"Bread",2,"slice",30.0}], tally_total:47.0}

[3] LLM → calculate_final(meal_time="midday", blood_glucose=140)
    → Harness validates: tally not empty → OK
    Tool → {final:4.13, food_insulin:3.13, glucose_correction:1.00,
            meal_time:"midday", meal_hour:null, ...}

[4] LLM → TEXT:
    "Potatoes: 100g → 17.0g carbs
     Bread: 2 slice → 30.0g carbs
     Total: 47.0g | Midday (divider 15)
     Food insulin: 47.0/15 = 3.13 U
     BG correction: (140-100)/40 = 1.00 U
     🎯 Final dose: 4.13 U"
```

**4 iterations** — no parallel complexity needed, no separate setter calls.

### 4.2 Ambiguity Handling

```
User: "I ate rice"

Harness injects:
  "I ate rice

   [CURRENT TALLY: empty]
   [KNOWN FOOD IDS: none]"

[1] LLM → search_foods(["rice"])
    Tool → [{id:5, name:"Rice (cooked)", ...}, {id:6, name:"Rice (raw)", ...}]
    → knownFoodIds updated to {5, 6}

[2] LLM → TEXT: "I found 'Rice (cooked)' and 'Rice (raw)'. Which one did you eat?"
```

```
User: "Cooked, 200g"

Harness injects:
  "Cooked, 200g

   [CURRENT TALLY: empty]
   [KNOWN FOOD IDS: Rice(5), Rice(6)]"    ← IDs from previous turn preserved

[3] LLM → add_foods_to_tally([{5, 200, "g"}])
    → Harness validates: 5 is in knownFoodIds → OK
    Tool → {entries: [{1,"Rice (cooked)",200,"g",56.0}], tally_total:56.0}

[4] LLM → calculate_final()
    Tool → {final:3.73, food_insulin:3.73, ...}
```

### 4.3 Correction Mid-Flow

```
User: "I ate 100g potatoes, 2 breads, BG 140."

Harness injects:
   "I ate 100g potatoes, 2 breads, BG 140.
   
    [CURRENT TALLY: empty]
    [KNOWN FOOD IDS: none]"

[1] LLM → search_foods(["potatoes", "bread"])
    → knownFoodIds = {12, 25}
[2] LLM → add_foods_to_tally([{12, 100, "g"}, {25, 2, "slice"}])
    Tool → tally_total: 47.0g, entries: [{1, "Potatoes", ...}, {2, "Bread", ...}]
[3] LLM → calculate_final(meal_time="midday", blood_glucose=140)
    Tool → {final:4.13, ...}
[4] LLM → TEXT: "Potatoes 17.0g + Bread 30.0g = 47.0g total. Midday/15=3.13U. BG +1.00U. Final: 4.13U"
```

```
User: "Wait, no bread actually."

Harness injects:
   "Wait, no bread actually.
   
    [CURRENT TALLY: 2 items, 47.0g total]
      Potatoes (boiled) 100g = 17.0g (entry_id: 1)
      Bread (white) 2slice = 30.0g (entry_id: 2)
    [KNOWN FOOD IDS: Potatoes(12), Bread(25)]"

[5] LLM → remove_foods_from_tally([2])     ← entry_id:2 visible in context
    → Harness validates: 2 exists in tally → OK
    Tool → {removed:1, tally_total:17.0}

[6] LLM → calculate_final(meal_time="midday", blood_glucose=140)
    Tool → {final:2.13, food_insulin:1.13, glucose_correction:1.00, ...}

[7] LLM → TEXT: "Bread removed. Potatoes only: 17g/15=1.13U. BG: +1.00U. Final: 2.13U"
```

### 4.4 Glucose-Only Check (No Foods)

```
User: "My blood sugar is 95"

Harness injects:
  "My blood sugar is 95

   [CURRENT TALLY: empty]
   [KNOWN FOOD IDS: none]"

[1] LLM → calculate_final(blood_glucose=95)
    → Harness validates: tally empty → ERROR "Add at least one food first"

[2] LLM → TEXT: "You haven't added any foods yet. To check glucose correction alone,
                  please add a food item first or use the manual Calculation tab."
```

### 4.5 Multi-Turn with State Persistence

```
User: "I ate 100g potatoes"

Harness injects:
  "I ate 100g potatoes
   [CURRENT TALLY: empty]
   [KNOWN FOOD IDS: none]"

[1] LLM → search_foods(["potatoes"])
    → knownFoodIds = {12}
[2] LLM → add_foods_to_tally([{12, 100, "g"}])
[3] LLM → TEXT: "Potatoes added: 17.0g carbs. What else did you eat?"
```

```
User: "also 2 breads"

Harness injects:
  "also 2 breads
   [CURRENT TALLY: 1 item, 17.0g total]
     Potatoes (boiled) 100g = 17.0g (entry_id: 1)
   [KNOWN FOOD IDS: Potatoes(12)]"

[4] LLM → search_foods(["bread"])
    → knownFoodIds = {12, 25}
[5] LLM → add_foods_to_tally([{25, 2, "slice"}])
    Tool → tally_total: 47.0g
[6] LLM → TEXT: "Bread added: 30.0g. Total now 47.0g. Ready to calculate?"
```

---

## 5. Fine-Tuning Strategy (Both Candidates)

The same 8,000-conversation dataset is used for both models. Only the chat template differs per model.

### 5.1 Training Dataset

**Size**: ~8,000 multi-turn conversations  
**Languages**: ~800 per language (balanced across 10)  
**Format**: Model-specific chat template

- **Candidate A (LFM2.5)**: ChatML format with `<|im_start|>system` / `<|im_start|>user` / `<|im_start|>assistant` / `<|im_start|>tool` roles, tool calls between `<|tool_call_start|>` / `<|tool_call_end|>` tokens
- **Candidate B (Gemma 4)**: Gemma chat template with `<start_of_turn>` role markers, JSON function call format

The same 8,000 conversation scenarios are rendered into both templates so each model trains on identical semantic content.

**Scenario distribution**:

| Type | % | Count |
|------|----|-------|
| Simple single food, no glucose | 15% | 1,200 |
| Multiple foods, no glucose | 20% | 1,600 |
| Multiple foods + glucose | 25% | 2,000 |
| With explicit meal time | 10% | 800 |
| Ambiguous food (clarification needed) | 10% | 800 |
| Correction/removal mid-flow | 8% | 640 |
| Only glucose check | 5% | 400 |
| Incomplete info (asks for more) | 5% | 400 |
| Food not found in DB | 2% | 160 |

### 5.2 Training Configuration

#### Candidate A: LFM2.5-1.2B-Instruct

```
Environment: 1x A100 40GB (or GCP L4)
Framework: Unsloth (LoRA/QLoRA SFT)
Base model: LiquidAI/LFM2.5-1.2B-Instruct

LoRA:
  rank: 16
  alpha: 32
  target_modules: all linear layers
  4-bit quantization (QLoRA via Unsloth)

Training:
  epochs: 3-5
  batch size: 4 (with gradient accumulation)
  learning rate: 2e-4
  warmup: 100 steps
  max sequence length: 2048

Time: ~4-6 hours on A100
```

#### Candidate B: Gemma 4 E2B QAT

```
Environment: 1x A100 40GB (or GCP L4)
Framework: HuggingFace Transformers + PEFT (QLoRA) or Unsloth
Base model: google/gemma-4-E2B-it-qat-mobile

QLoRA:
  rank: 16
  alpha: 32
  target_modules: all linear layers
  4-bit quantization (NF4)

Training:
  epochs: 3-5
  batch size: 2 (smaller — larger model)
  learning rate: 2e-4
  warmup: 100 steps
  max sequence length: 2048

Time: ~6-8 hours on A100
```

### 5.3 Data Generation Pipeline

Use Gemini 2.5 Pro API to generate synthetic conversations:
1. Feed all 6 tool schemas + calculation formulas + food DB samples
2. For each language → generate 800 diverse conversations
3. Include edge cases: ambiguity, corrections, missing foods
4. Validate by running through mock harness
5. Export in **both** chat template formats (LFM2.5 ChatML + Gemma 4 Gemma format)

### 5.4 Post-Training

1. Merge LoRA weights into base model
2. Evaluate on test set (target: >85% correct tool call sequences for both candidates)
3. Quantize for deployment:
   - Candidate A → GGUF Q4_0 (~719MB) via llama.cpp quantize tool
   - Candidate B → `.litertlm` mobile QAT format (<1GB) via LiteRT-LM converter
4. Test on physical devices (Samsung S25, Pixel)
5. Run model selection criteria (see Section 10.5)

---

## 6. Android Integration

The specific dependencies depend on which model wins the evaluation. The Kotlin `ai/` package is designed to be model-agnostic — only `LlmInferenceEngine.kt` has model-specific implementations.

### 6.1 Shared Dependencies (Both Paths)

```kotlin
// app/build.gradle.kts — always needed
implementation("com.google.android.play:asset-delivery:2.3.0")
implementation("com.google.android.play:asset-delivery-ktx:2.3.0")
implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0")
```

### 6.2 Path A — If LFM2.5-1.2B Wins

```kotlin
// app/build.gradle.kts
// Option 1: llama.cpp Android bindings (GGUF CPU inference)
// implementation("com.github.ggerganov:llama.cpp:latest")
// Option 2: LEAP SDK (Liquid's platform — includes NPU acceleration)
// implementation("ai.liquid:leap-android:latest")
```

**Model format**: GGUF (Q4_0 quantized, ~719MB)  
**Runtime**: llama.cpp via JNI/Kotlin bindings, or LEAP SDK for NPU acceleration  
**Delivery**: Play Asset Delivery (install-time), fallback download  
**NPU option**: Qualcomm NexaML integration (82 tok/s on ROG Phone9 Pro)

### 6.3 Path B — If Gemma 4 E2B QAT Wins

```kotlin
// app/build.gradle.kts
implementation("com.google.ai.edge.litertlm:litertlm-android:1.0.0-beta02")
```

**Model format**: `.litertlm` (mobile QAT, <1GB)  
**Runtime**: LiteRT-LM SDK (Google first-party)  
**Delivery**: Play Asset Delivery (install-time), fallback download

### 6.4 Model-Agnostic File Structure

All code under `ai/` is model-agnostic:

```
app/src/main/java/com/kgiantsios/mininiocarbtracker/
├── ai/
│   ├── AgenticHarness.kt        # ReAct loop — model-agnostic
│   ├── AgenticState.kt          # State machine
│   ├── ToolExecutor.kt          # Adapter to DietViewModel
│   ├── ToolSchemas.kt           # Static JSON schema definitions
│   ├── LlmInferenceEngine.kt    # Interface (abstraction)
│   ├── LfmInferenceEngine.kt    # Impl A: wraps llama.cpp, parses <|tool_call_start|>
│   ├── GemmaInferenceEngine.kt  # Impl B: wraps LiteRT-LM, parses Gemma function JSON
│   ├── ModelManager.kt          # Download, verify, cache model — same for both
│   └── AiChatViewModel.kt       # ViewModel for chat UI state
├── ui/
│   ├── AiAssistScreen.kt        # Tab 4: full chat interface
│   └── ai/
│       ├── AiChatMessage.kt     # Chat bubble composable
│       ├── AiToolCallBubble.kt  # Tool call progress indicator
│       └── AiInputBar.kt        # Text input + send button
```

**The `LlmInferenceEngine` interface**:
```kotlin
interface LlmInferenceEngine {
    suspend fun generate(
        messages: List<ChatMessage>,
        tools: List<ToolSchema>
    ): LlmResponse  // sealed class: TextResponse | ToolCallsResponse(calls: List<ToolCall>)
    
    val modelInfo: ModelInfo  // name, version, size
}
```

Both implementations produce the same `LlmResponse` types. The harness doesn't care which engine is underneath.

### 6.5 Model Delivery

Play Asset Delivery (install-time) for both paths:
- Module: `carb_model`
- Asset for Candidate A: `models/carb-calc-finetune.gguf` (~719MB)
- Asset for Candidate B: `models/carb-calc-finetune.litertlm` (<1GB)
- Fallback: download on first AI use if PAD unavailable

### 6.6 Cross-Tab Synchronization

The AI tab and Calculation tab share the same `DietViewModel`:
- AI adds foods → switch to Calc tab → see them in AddedItemsCard
- Manual additions → switch to AI tab → "what's my tally?" shows them
- Same ResultCard composable reused in both tabs
- Same Save-to-History flow

---

## 7. UI — "AI Assist" Tab (Tab 4)

### Layout
```
┌──────────────────────────────────────────────┐
│  Top Bar: "AI Assist"                        │
├──────────────────────────────────────────────┤
│  Welcome card (first visit only)              │
│  ─────────────────────────────────────        │
│  Chat messages (scrollable LazyColumn)        │
│  ├─ User bubbles (right-aligned, Rhino Teal)  │
│  ├─ AI text bubbles (left-aligned)            │
│  ├─ Tool call cards (collapsible)             │
│  └─ Result card (reused from CompScreen)      │
│  ─────────────────────────────────────        │
│  Text input field + Send button               │
└──────────────────────────────────────────────┘
```

### Bottom Navigation Bar
`[Home] [Calculate] [History] [Nutrition] [AI Assist]`

### Key UI behaviors
- Auto-scroll to latest message
- Tool calls as collapsible cards (tap to see JSON)
- Shimmer typing indicator while LLM thinks
- "Stop" button during generation
- Long "thinking" (>3s) → "Still processing..." message

---

## 8. Error Handling

| Scenario | Behavior |
|----------|----------|
| Food not in DB | LLM asks: "I couldn't find [X]. Try another name." |
| Multiple matches | LLM asks: "Did you mean [A] or [B]?" |
| Missing quantity | LLM asks: "How much [food] did you eat?" |
| LLM exceeds 15 iterations | Force-stop: "Please try manual calculation." |
| Model not downloaded | Progress bar with size. Manual calc works fine. |
| Low memory (<2GB) | Warning: "AI may be slow. Manual calc always works." |
| LLM hallucinates food ID | ToolExecutor validates → error → retry |
| No network | Model is local (PAD). No network needed for inference. |

---

## 9. Settings

New settings in SettingsScreen → AI Assist section:

| Setting | Default | Description |
|---------|---------|-------------|
| AI Assist Enabled | ON | Show/hide the AI tab |
| Model Auto-Download | Wi-Fi only | When to download model (~700MB-1GB) |
| Delete Model | — | Free storage |
| AI Temperature | 0.1 | Determinism vs creativity |

New DataStore keys in SettingsRepository:
- `ai_assist_enabled` (Boolean)
- `ai_model_downloaded` (Boolean)
- `ai_temperature` (Double)

---

## 10. Implementation Phases

| Phase | Deliverable | Est. Time |
|-------|-------------|-----------|
| 1. Tool plumbing | ToolSchemas, ToolExecutor, mock harness, AgenticState, AiChatViewModel | 1 week |
| 2a. UI | AiAssistScreen, AiChatMessage, AiInputBar, AiToolCallBubble, bottom nav | 1 week |
| 2b. Cross-tab sync | Shared DietViewModel, result card reuse | 0.5 week |
| 3. Data generation | Python script + Gemini API → 8,000 conversations in both chat templates | 1 week |
| 4a. Fine-tune Candidate A | LoRA on LFM2.5-1.2B-Instruct, 4-6h A100 | 0.5 week |
| 4b. Fine-tune Candidate B | QLoRA on Gemma 4 E2B QAT, 6-8h A100 | 0.5 week |
| 5. Model selection | Evaluate both, apply criteria (Sec 10.5), pick winner | 0.5 week |
| 6. Android ML | Integrate winning runtime (llama.cpp or LiteRT-LM), ModelManager, PAD | 1 week |
| 7. Wire up | Replace mock with real LLM, streaming, error handling | 0.5 week |
| 8. Polish | Settings, model management, i18n, accessibility, testing | 1 week |
| **Total** | | **~7.5 weeks** |

### 10.5 Model Selection Criteria

After fine-tuning both candidates, evaluate on a held-out test set of 500 conversations (50 per language). Score each model:

| Metric | Weight | How measured | Target |
|--------|--------|-------------|--------|
| **Tool call accuracy** | 40% | % of individual tool calls with correct name **and** correct parameters | >85% |
| **Sequence correctness** | 25% | % of complete conversations ending with correct `final_result` (within 1% tolerance) | >90% |
| **Clarification quality** | 15% | % of ambiguous-food scenarios where model asks the right clarifying question | >80% |
| **Natural language quality** | 10% | Human rating (1-5) of conversation fluency, averaged across all 10 languages | >3.5 |
| **Latency** | 5% | Time-to-first-token + tokens/sec on reference device (Samsung S25 Ultra) | TTFT <1s |
| **Memory footprint** | 5% | Peak RAM during 10-conversation stress test | <1.5GB |

**Scoring formula**:
```
score = (tool_accuracy × 0.40) + (sequence_correctness × 0.25) + (clarification × 0.15)
      + (language_quality/5 × 0.10) + (latency_normalized × 0.05) + (memory_normalized × 0.05)
```

**Tiebreaker**: If scores are within 5 percentage points, prefer the model with **no license revenue cap** (Candidate B: Gemma 4 E2B QAT).

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| LFM2.5 underperforms on tool calling after FT | Candidate B (Gemma 4) is the fallback — dual evaluation catches this early |
| Gemma 4 fails to learn tool calling from scratch | Candidate A (LFM2.5) has pre-existing tool calling baseline — lower risk path |
| LFM2.5 license becomes a blocker ($10M cap) | Gemma 4 has no revenue cap; always available as migration path |
| Both models underperform on 4 minority languages | Add more training data for el/hi/it/pt; use rule-based fallback for those languages |
| llama.cpp Android bindings are unstable | LEAP SDK as alternative; Gemma 4 + LiteRT-LM as fallback path |
| Model size (~700MB-1GB) rejected by users | Optional feature; manual calc always works; clear storage info |
| LiteRT-LM API unstable | Pin version; used by Google's own Edge Gallery app |
| LLM makes math errors | ToolExecutor validates all inputs; same DietCalculator logic runs (no AI math) |

---

## 12. Key Architectural Principles

1. **The LLM is a smart input parser, not a calculator.** All math goes through `DietCalculator.kt`.
2. **Single source of truth.** `DietUiState` is shared between AI and manual flows.
3. **Offline-first.** Model is bundled via Play Asset Delivery — no network needed.
4. **Graceful degradation.** Manual calculation always works, with or without AI.
5. **ToolExecutor is a thin adapter.** It maps LLM function calls to existing `DietViewModel` methods.
