"""Improvement orchestration layer for AI-Ready.

This package integrates Apache Burr as a stateful workflow engine
for managing knowledge improvement lifecycles. It sits on top of
AI-Ready's existing scanning and assessment pipeline.

Architecture:
  models.py       — Improvement state, status, hypothesis, proposal, verification
  actions.py      — Burr actions (analyze, propose, approve, execute, verify)
  application.py  — Burr ApplicationBuilder graph with transitions
  executor.py     — AI-Ready's execution layer for applying knowledge modifications
  forking.py      — State forking for retrying failed improvements
  history.py      — Improvement history store (institutional memory)
  test_generation.py — Regression test generation from failed improvements
  manager.py      — ImprovementManager facade tying it all together
  parsing.py      — JSON extraction/repair utilities for LLM responses
  prompts.py      — Prompt context builders (signal, artifact, history, assessment)
  heuristics.py   — Heuristic fallbacks, proposal filtering, LLM response parsing
  salience.py     — Signal-delta gating, problem discovery, salience ranking
  diagnosis_quality.py — Rolling correlation for diagnosis quality tracking

Flow:
  Assessment Report → Burr Improvement Workflow → Verified Knowledge Changes

Burr manages: state transitions, persistence, tracking, replay/forking
AI-Ready manages: scanning, assessment, execution, verification
LLM provides: reasoning inside Burr actions (root cause analysis, proposals)
"""
