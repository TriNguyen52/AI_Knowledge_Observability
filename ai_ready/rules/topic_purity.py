"""Topic Purity collector — detect artifacts containing multiple unrelated concepts."""

from __future__ import annotations

import re
import math
from collections import Counter
from typing import Any

from ai_ready.models import ArtifactBundle, CollectorResult, KnowledgeSignal, Severity
from ai_ready.rules import SignalCollector, register_collector


@register_collector
class TopicPurityCollector(SignalCollector):
    """Detect artifacts that mix unrelated topics via heading clustering."""

    id = "topic_purity"
    severity_default = Severity.HIGH
    dimension = "retrieval"

    # Stopwords for keyword extraction
    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "up", "down", "out", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how",
        "all", "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "just", "also", "and", "or", "but", "if", "while", "about",
        "this", "that", "these", "those", "it", "its", "they", "them",
        "their", "we", "us", "our", "you", "your", "he", "she", "his", "her",
    }

    def collect(self, bundle: ArtifactBundle) -> dict[str, Any]:
        """Extract heading groups and their keyword sets from each artifact."""
        doc_topics: list[dict[str, Any]] = []
        for artifact in bundle.artifacts:
            if len(artifact.headings) < 2:
                doc_topics.append({"doc_id": artifact.id, "doc_path": artifact.uri, "clusters": [], "title": artifact.title})
                continue

            # Determine the minimum heading level (the top-level heading)
            min_level = min(h.level for h in artifact.headings)

            # Group headings by top-level heading
            clusters: list[dict[str, Any]] = []
            current_cluster: dict[str, Any] | None = None

            for heading in artifact.headings:
                if heading.level == min_level:
                    if current_cluster:
                        clusters.append(current_cluster)
                    current_cluster = {
                        "heading": heading.text,
                        "keywords": self._extract_keywords(heading.text),
                        "sub_headings": [],
                    }
                elif current_cluster:
                    current_cluster["sub_headings"].append(heading.text)
                    current_cluster["keywords"].update(self._extract_keywords(heading.text))

            if current_cluster:
                clusters.append(current_cluster)

            doc_topics.append({
                "doc_id": artifact.id,
                "doc_path": artifact.uri,
                "title": artifact.title,
                "clusters": clusters,
            })
        return {"doc_topics": doc_topics}

    def measure(self, signals: dict[str, Any]) -> dict[str, Any]:
        """Compute cluster count, largest cluster ratio, and topic entropy per artifact."""
        results: list[dict[str, Any]] = []
        for doc_info in signals["doc_topics"]:
            clusters = doc_info["clusters"]
            cluster_count = len(clusters)

            if cluster_count <= 1:
                results.append({
                    "doc_id": doc_info["doc_id"],
                    "doc_path": doc_info["doc_path"],
                    "title": doc_info["title"],
                    "cluster_count": cluster_count,
                    "largest_cluster_ratio": 1.0,
                    "topic_entropy": 0.0,
                    "clusters": clusters,
                })
                continue

            # Compute keyword overlap between clusters
            all_keywords: set[str] = set()
            for c in clusters:
                all_keywords.update(c["keywords"])

            # Compute similarity matrix
            cluster_keywords = [c["keywords"] for c in clusters]
            similarities: list[float] = []
            for i in range(len(cluster_keywords)):
                for j in range(i + 1, len(cluster_keywords)):
                    sim = self._jaccard(cluster_keywords[i], cluster_keywords[j])
                    similarities.append(sim)

            # Largest cluster ratio: size of biggest cluster / total
            cluster_sizes = [len(c["keywords"]) + 1 for c in clusters]
            total = sum(cluster_sizes)
            largest_ratio = max(cluster_sizes) / total if total > 0 else 1.0

            # Topic entropy: how evenly distributed are clusters
            probs = [s / total for s in cluster_sizes if s > 0]
            entropy = -sum(p * math.log2(p) for p in probs) if probs else 0.0

            results.append({
                "doc_id": doc_info["doc_id"],
                "doc_path": doc_info["doc_path"],
                "title": doc_info["title"],
                "cluster_count": cluster_count,
                "largest_cluster_ratio": largest_ratio,
                "topic_entropy": entropy,
                "avg_similarity": sum(similarities) / len(similarities) if similarities else 0.0,
                "clusters": clusters,
            })

        return {"doc_metrics": results}

    def evaluate(self, metrics: dict[str, Any]) -> list[KnowledgeSignal]:
        """Flag artifacts with multiple unrelated clusters.

        Emits bare signals with signal_type only. Severity, score, and
        recommendation are assigned by the InterpretationPolicy in the pipeline.
        """
        signals: list[KnowledgeSignal] = []
        for dm in metrics["doc_metrics"]:
            if dm["cluster_count"] > 1 and dm["largest_cluster_ratio"] < 0.7:
                cluster_names = [c["heading"] for c in dm["clusters"]]
                signals.append(KnowledgeSignal(
                    collector_id=self.id,
                    signal_type="mixed_topics",
                    severity=Severity.LOW,  # Placeholder; overridden by policy
                    score=0,  # Placeholder; overridden by policy
                    artifact_id=dm["doc_id"],
                    artifact_uri=dm["doc_path"],
                    evidence={
                        "cluster_count": dm["cluster_count"],
                        "largest_cluster_ratio": round(dm["largest_cluster_ratio"], 3),
                        "topic_entropy": round(dm["topic_entropy"], 3),
                        "clusters": [[c["heading"]] + c.get("sub_headings", []) for c in dm["clusters"]],
                    },
                    recommendation="",  # Filled by policy
                ))
        return signals

    def report(self) -> CollectorResult:
        """Produce final result.

        Score and severity are placeholders; the pipeline recomputes them
        from policy-assessed signals during aggregation.
        """
        signals = self._findings
        metrics = self._metrics

        all_docs = metrics.get("doc_metrics", [])
        total_entropy = sum(d.get("topic_entropy", 0) for d in all_docs)
        avg_entropy = total_entropy / len(all_docs) if all_docs else 0

        score = max(0, 100 - len(signals) * 15 - int(avg_entropy * 10))
        severity = Severity.HIGH if signals else Severity.LOW

        return CollectorResult(
            collector_id=self.id,
            score=score,
            severity=severity,
            metrics={
                "total_documents": len(all_docs),
                "flagged_documents": len(signals),
                "avg_topic_entropy": round(avg_entropy, 3),
            },
            signals=signals,
            recommendation=f"{len(signals)} artifacts have mixed topics" if signals else "All artifacts are topically focused",
        )

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract meaningful keywords from text."""
        words = re.findall(r"[a-zA-Z]{2,}", text.lower())
        return {w for w in words if w not in self.STOPWORDS}

    def _jaccard(self, a: set[str], b: set[str]) -> float:
        """Jaccard similarity between two sets."""
        if not a and not b:
            return 1.0
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)
