# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resolves ontology terms by searching the AlphaGenome ontology mapping file for closest matches.

This script acts as a candidate retriever for mapping free-text queries to
AlphaGenome-compatible ontology terms. It performs a fast,
simple word-overlap search over the available tissues and returns the top
matches ranked by score.

Design Note:
  This tool acts as a simple candidate retriever for ontology mapping.
  It does not perform complex internal synonym mapping or semantic
  rules, leaving the calling agent (or researcher) to leverage their
  domain knowledge for query expansion (e.g., "cardiac" -> "heart")
  and final candidate selection.

Usage:
  uv run resolve_ontology_terms.py --query='liver'

Examples:
  uv run resolve_ontology_terms.py --query='liver'
  uv run resolve_ontology_terms.py --query='k562' --limit=5
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "alphagenome",
#   "pandas",
#   "python-dotenv",
# ]
# ///

import argparse
import difflib
import json
import os
import re
import sys
from typing import Any, Sequence

import dotenv

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RESOURCES_DIR = os.path.join(_SCRIPT_DIR, "..", "resources")
_MAPPING_FILE = os.path.join(_RESOURCES_DIR, "tissue_ontology_mapping.json")
_FUZZY_RATIO_THRESHOLD = 0.7
_KEYWORD_PREFIX_LENGTH = 5


def normalize_and_split(text: str) -> set[str]:
  """Lowercases and splits text into alphanumeric words of length > 2."""
  text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
  return {t.strip() for t in text.split() if len(t.strip()) > 2}


def search_ontology(
    query: str, mapping: dict[str, Any], limit: int = 10
) -> list[dict[str, Any]]:
  """Search for tissues matching the query words."""
  query_words = normalize_and_split(query)

  if not query_words:
    return []

  results = []
  for curie, data in mapping.items():
    name = data.get("biosample", {}).get("name", "")
    # Flatten all metadata for searching
    full_text = f"{curie} {name} {str(data)}"
    record_words = normalize_and_split(full_text)

    # Calculate match score with fuzzy matching
    score = 0.0
    name_words = normalize_and_split(name)
    for qw in query_words:
      if qw in record_words:
        score += 1.0
        if qw in name_words:
          score += 2.0
      else:
        # Match if fuzzy similarity exceeds threshold or shares common prefix
        for rw in record_words:
          matcher = difflib.SequenceMatcher(isjunk=None, a=qw, b=rw)
          longest = matcher.find_longest_match(0, len(qw), 0, len(rw))
          shares_prefix = (
              longest.size >= _KEYWORD_PREFIX_LENGTH
              and longest.a == 0
              and longest.b == 0
          )
          if matcher.ratio() >= _FUZZY_RATIO_THRESHOLD or shares_prefix:
            score += 0.5
            if rw in name_words:
              score += 1.0
            break

    if score > 0:
      results.append({
          "curie": curie,
          "name": name,
          "type": data.get("biosample", {}).get("type", "N/A"),
          "assays": list(data.get("assays", {}).keys()),
          "score": score,
      })

  # Sort by score descending, then by name length
  # (prefer shorter, more specific names if score tied)
  results.sort(key=lambda x: (x["score"], -len(x["name"])), reverse=True)
  return results[:limit]


def main(argv: Sequence[str] | None = None) -> None:
  dotenv.load_dotenv(os.path.expanduser("~/.env"))
  parser = argparse.ArgumentParser(
      description="Resolves ontology terms by searching available tracks."
  )
  parser.add_argument(
      "--query", required=True, help="Tissue name or keyword to search for."
  )
  parser.add_argument(
      "--limit", type=int, default=10, help="Max number of results to return."
  )
  parser.add_argument(
      "--mapping_path",
      default=_MAPPING_FILE,
      help="Path to tissue_ontology_mapping.json.",
  )
  args = parser.parse_args(argv)

  mapping_path = args.mapping_path
  if not os.path.exists(mapping_path):
    print(
        f"Error: Mapping file not found at {mapping_path}.",
        file=sys.stderr,
    )
    print(
        "Please run generate_ontology_mapping.py first to generate it.",
        file=sys.stderr,
    )
    return

  with open(mapping_path, "r") as f:
    mapping = json.load(f)

  results = search_ontology(args.query, mapping, args.limit)

  print(f"\nSearch results for: '{args.query}'")
  print(f"Query words used: {normalize_and_split(args.query)}")
  print("-" * 60)

  if not results:
    print("No matches found.")
  else:
    print(
        f"{'Rank':<4} | {'ID':<15} | {'Name':<40} | {'Type':<10} | {'Score':<5}"
    )
    print("-" * 80)
    for i, res in enumerate(results):
      # Truncate name if too long for table
      name = res["name"][:37] + "..." if len(res["name"]) > 40 else res["name"]
      print(
          f"[{i+1:<2}] | {res['curie']:<15} | {name:<40} | {res['type']:<10} |"
          f" {res['score']:.1f}"
      )


if __name__ == "__main__":
  main()
