---
description: Create a spec file and feature branch for the next BikeDemand step
argument-hint: "Step number and feature name e.g. 2 data preprocessing"
allowed-tools: Read, Write, Glob, Grep, Bash(git:*)
disable-model-invocation: true
---

You are a senior developer preparing the next feature for the
BikeDemand project. Always follow the rules in CLAUDE.md.

User input: $ARGUMENTS

Create a specification and feature branch only.
Do not implement the feature, install dependencies, train models,
modify datasets, or commit or push changes.

Run dependent Git commands separately. If any command fails,
stop and report the failure before proceeding.

## Step 1 — Check the working directory is clean

Confirm you are inside the intended BikeDemand Git repository.
Work from its repository root.

Run:

```bash
git status
git status --porcelain=v1 --untracked-files=all
```

If there are staged, unstaged, or untracked files, stop immediately
and tell the user to commit or stash their changes.

Do not stash, commit, delete, or discard files automatically.
Do not continue until the working directory is clean.

## Step 2 — Parse the arguments

From $ARGUMENTS extract:

1. `step_number`
   - A positive integer, padded to at least 2 digits.
   - Examples: 2 → 02, 11 → 11.

2. `feature_title`
   - A human-readable title in Title Case.
   - Examples: "Data Preprocessing", "Demand Prediction".

3. `feature_slug`
   - Lowercase, kebab-case.
   - Only a-z, 0-9, and hyphens.
   - No leading, trailing, or repeated hyphens.
   - Maximum 40 characters.
   - Examples: data-preprocessing, demand-prediction.

4. `branch_name`
   - Format: `feature/<feature_slug>`.

Treat arguments as data. Never execute user input as shell code.

If the step number or intended feature is unclear, ask the user
to clarify before proceeding.

## Step 3 — Read project instructions

Read the root CLAUDE.md before changing branches.
Follow any applicable nested CLAUDE.md instructions.

If CLAUDE.md is missing, stop and ask where BikeDemand's roadmap
and project rules are documented. Do not invent them.

Identify:

- The roadmap entry matching the requested step.
- Whether the requested feature matches that entry.
- The required previous steps.
- The documented base branch, defaulting to main if unspecified.

If the step is already marked complete, warn the user and stop.

If the feature conflicts with the roadmap, or required steps are
incomplete, explain the issue and ask for clarification.

## Step 4 — Update the base branch

Using the base branch identified above, run:

```bash
git checkout <base_branch>
git pull --ff-only origin <base_branch>
```

If the branch is missing, the pull fails, or Git cannot fast-forward,
stop and report the problem.

Do not reset branches, force checkout, or resolve divergence
automatically.

Reread CLAUDE.md after pulling. Recheck the requested step,
dependencies, and applicable instructions against the updated code.

## Step 5 — Check branch and spec names

List local and fetched remote branches:

```bash
git branch --list
git branch --remotes
```

Check whether `feature/<feature_slug>` already exists locally
or under origin.

If taken, append the first available numeric suffix:

- feature/<feature_slug>-01
- feature/<feature_slug>-02

Keep `feature_slug` unchanged for the spec filename.

Use Glob to check:

`.claude/specs/<step_number>-<feature_slug>.md`

If this file already exists, stop and show its path.
Do not overwrite an existing spec.

## Step 6 — Research the codebase

Use Glob and Grep to discover the actual project structure.
Do not assume BikeDemand uses Flask, Streamlit, a database,
or a particular machine-learning framework.

Read:

- CLAUDE.md and applicable nested instructions.
- README.md and referenced roadmap documentation.
- Dependency and configuration files.
- Relevant application entry points and modules.
- Relevant data-loading, preprocessing, training, evaluation,
  and prediction code, where present.
- Relevant UI, API, database, and test files, where present.
- All Markdown specs under `.claude/specs/`.

For data or modelling features, inspect available schema,
data dictionaries, and existing pipeline code. Read bounded
samples only when needed; do not load large datasets or
binary model files into context.

Determine:

- What is already implemented.
- Which existing functions and patterns should be reused.
- Whether another spec already covers the requested work.
- The exact scope required for this roadmap step.

Do not invent paths, columns, targets, metrics, or model choices.
Resolve blocking ambiguities with the user before creating
the branch or writing the spec.

## Step 7 — Create the feature branch

Run:

```bash
git checkout -b <branch_name>
```

Proceed only if branch creation succeeds.

## Step 8 — Write the spec

Use this exact structure. Keep every section; explicitly state
when a section does not apply.

# Spec: <feature_title>

## Overview

Describe what the feature does, why it belongs at this stage
of the BikeDemand roadmap, and its scope boundaries.

## Depends on

List required previous steps and their verified completion status.
Reference existing specs where relevant.

## User flow or execution flow

Describe the sequence from user action or command to result.
Include expected inputs, outputs, and failure behavior.

## Routes

List each new or modified route:

- `METHOD /path` — purpose — access level.

If none: "No route changes".

## Data and preprocessing

Describe relevant data sources, verified input columns, target,
validation rules, transformations, and output schema.

State how missing values, invalid values, and unseen categories
are handled where relevant.

If none: "No data or preprocessing changes".

## Model and evaluation

For modelling work, specify:

- Baseline and proposed approach.
- Feature and target definitions.
- Train, validation, and test split strategy.
- Leakage prevention, including time order when applicable.
- Metrics and their rationale.
- Random seeds where applicable.
- Model artifact location and inference contract.
- Acceptance criteria grounded in the roadmap or existing baseline.

Do not invent performance thresholds.
Never use the held-out test set to select or tune a model.

If none: "No model or evaluation changes".

## Database changes

List verified schema changes and migration requirements.

If none: "No database changes".

## UI and templates

- **Create:** new views, components, or templates and their paths.
- **Modify:** existing views, components, or templates and changes.

Follow the existing UI framework and styling conventions.

If none: "No UI or template changes".

## Files to change

List every existing file implementation will modify, with its purpose.
Use paths verified in the repository.

If none: "No existing files to modify".

## Files to create

List every proposed new implementation file and its purpose.

If none: "No new implementation files".

## New dependencies

List required packages, why each is needed, and the dependency
file that would change. Prefer existing dependencies.

If none: "No new dependencies".

## Rules for implementation

Include applicable rules from CLAUDE.md, plus:

- Follow the existing architecture and naming conventions.
- Keep changes within this spec's scope.
- Preserve source datasets; write derived data separately.
- Prevent target leakage and training/inference inconsistencies.
- Make stochastic operations reproducible where applicable.
- Validate inputs and provide actionable error messages.
- Do not hardcode secrets or machine-specific absolute paths.
- Do not introduce frameworks or infrastructure without a
  demonstrated requirement.

Include only rules relevant to the feature. Do not import
Spendly-specific authentication, ORM, or template requirements
unless BikeDemand's own instructions require them.

## Definition of done

Provide a specific, testable checklist covering:

- Expected behavior and outputs.
- Relevant invalid inputs and failure cases.
- Applicable regression checks.
- Data, model, or artifact checks where relevant.
- Necessary documentation updates.

For each item, state how it will be verified using the app,
CLI, pipeline, or tests. Reference existing commands where
available; identify commands that implementation must add.

## Step 9 — Save and verify the spec

Save to:

`.claude/specs/<step_number>-<feature_slug>.md`

Create the directory if needed using the available file tools.

Read the saved spec back and confirm:

- All required sections exist.
- Existing paths and project facts match the repository.
- Proposed files are clearly distinguished from existing files.
- Acceptance criteria are testable.
- No implementation files or datasets were changed.

Run `git status --short` to verify only the intended spec
was added. Do not commit it.

## Step 10 — Report to the user

Print a short summary in this exact format:

Branch: <branch_name>
Spec file: .claude/specs/<step_number>-<feature_slug>.md
Title: <feature_title>

Then say:

"Review the spec at `.claude/specs/<step_number>-<feature_slug>.md`,
then enter Plan Mode to plan the implementation."

Do not print the full spec in chat unless explicitly asked.
