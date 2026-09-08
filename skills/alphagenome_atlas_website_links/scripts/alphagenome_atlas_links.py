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

"""AlphaGenome Atlas Deep-Link & Track Predictions Generator CLI.

Constructs validated deep-links and multi-modality Ref vs. Alt track predictions
for the AlphaGenome Atlas web application
(https://deepmind.google.com/science/alphagenome/atlas).

Subcommands:
  - `variant`: Construct single-variant exploration URLs.
  - `locus`: Construct genomic interval / locus exploration URLs.
  - `table`: Format variant records into a Markdown table with embedded Atlas
  links.
  - `track-predictions`: Construct /atlas/track-predictions comparison chart
  URLs with paired RNA-seq + splicing.
"""

import argparse
from collections.abc import Mapping, Sequence
import dataclasses
import json
import os
import re
import sys
import urllib.parse

from alphagenome.atlas import atlas
from alphagenome.data import genome
import dotenv
import pandas as pd

BASE_ATLAS_URL = "https://deepmind.google.com/science/alphagenome/atlas"
_DEFAULT_INTERGENIC_FLANK_BP = 20_000


@dataclasses.dataclass(frozen=True, kw_only=True)
class VariantLinkConfig:
  """Configuration for a single-variant Atlas URL.

  Attributes:
    variant: Genetic variant identifier in `chr:pos:ref>alt` format.
    biosample: Optional biosample or cell line name filter.
    modalities: Modality layout sections to display in the Atlas UI.
    filter_modalities: Scorer modality filters applied to the query (`f=`).
    tfs: Transcription factor names for ChIP-TF track filtering.
    include_avi: Whether to include the top-level AVI score track.
    extra_pinned: Additional track IDs or tokens to pin into the layout.
  """

  variant: str
  biosample: str | None = None
  modalities: Sequence[str] = ("RNA_SEQ", "DNASE", "CHIP_TF")
  filter_modalities: Sequence[str] = ()
  tfs: Sequence[str] = ()
  include_avi: bool = True
  extra_pinned: Sequence[str] = ()


@dataclasses.dataclass(frozen=True, kw_only=True)
class LocusLinkConfig:
  """Configuration for a locus/interval Atlas URL.

  Attributes:
    interval: Genomic coordinate viewport in `chr:start-end` format.
    biosample: Optional biosample or cell line name filter.
    modalities: Modality layout sections to display in the Atlas UI.
    filter_modalities: Scorer modality filters applied to the query (`f=`).
    tfs: Transcription factor names for ChIP-TF track filtering.
    include_avi: Whether to include the top-level AVI score track.
  """

  interval: str
  biosample: str | None = None
  modalities: Sequence[str] = ("RNA_SEQ", "DNASE", "CHIP_TF")
  filter_modalities: Sequence[str] = ()
  tfs: Sequence[str] = ()
  include_avi: bool = True


@dataclasses.dataclass(frozen=True, kw_only=True)
class GenomicInterval:
  """A genomic coordinate interval.

  Attributes:
    chrom: Chromosome with 'chr' prefix.
    start: 1-based start position (inclusive).
    end: 1-based end position (inclusive).
  """

  chrom: str
  start: int
  end: int

  def __str__(self) -> str:
    return f"{self.chrom}:{self.start}-{self.end}"


@dataclasses.dataclass(frozen=True, kw_only=True)
class GeneInfo:
  """Metadata for a single gene overlapping a genomic region.

  Attributes:
    gene_name: HGNC gene symbol.
    gene_id: Ensembl gene ID (with version).
    strand: Strand ('+' or '-').
    is_mane: Whether this is a MANE Select transcript.
    is_coding: Whether this is protein-coding.
    start: Gene start position.
    end: Gene end position.
  """

  gene_name: str
  gene_id: str
  strand: str
  is_mane: bool
  is_coding: bool
  start: int
  end: int


@dataclasses.dataclass(frozen=True, kw_only=True)
class ScoreRecord:
  """A single track prediction score entry.

  Attributes:
    modality: Scorer modality (e.g. 'RNA_SEQ', 'DNASE').
    track: Resolved track name from the Atlas catalog.
    strand: Strand descriptor for display.
    gene: Gene identifier or 'None' for unstranded modalities.
    score_id: Full score token for the Atlas URL.
  """

  modality: str
  track: str
  strand: str
  gene: str
  score_id: str


def normalize_variant_str(variant_str: str) -> str:
  """Validates and formats variant into chr:pos:ref>alt."""
  v = variant_str.strip()
  if not v.startswith("chr"):
    v = f"chr{v}"
  if ">" not in v:
    raise ValueError(
        f"Variant '{variant_str}' must contain '>' (format: chr:pos:ref>alt)"
    )
  return v


def encode_variant_query(variant_str: str) -> str:
  """URL-encodes variant for Atlas query string (ref%3Ealt)."""
  norm = normalize_variant_str(variant_str)
  return norm.replace(">", "%3E")


def _clean_chrom(chrom_str: str | int) -> str:
  """Ensures chromosome string has chr prefix."""
  c = str(chrom_str).strip()
  return c if c.startswith("chr") else f"chr{c}"


def parse_variant_components(variant: str) -> genome.Variant:
  """Parses a variant string (e.g. chr1:12345:A>G) into a genome.Variant."""
  return genome.Variant.from_str(variant)


def parse_interval(interval: str) -> GenomicInterval:
  """Parses a chr:start-end string into a GenomicInterval."""
  m = re.fullmatch(r"(?:chr)?([0-9A-Za-z]+):([0-9,]+)-([0-9,]+)", interval)
  if not m:
    raise ValueError(
        f"Invalid interval format '{interval}'. Expected 'chr:start-end'."
    )
  return GenomicInterval(
      chrom=_clean_chrom(m.group(1)),
      start=int(m.group(2).replace(",", "")),
      end=int(m.group(3).replace(",", "")),
  )


def _select_mane_preferred(df_rows):
  """Returns the MANE_Select-tagged row if available, otherwise the first row."""
  mane = df_rows[df_rows["tag"].fillna("").str.contains("MANE_Select")]
  return mane.iloc[0] if not mane.empty else df_rows.iloc[0]


def _parse_modality_list(modalities: Sequence[str] | str) -> list[str]:
  """Normalizes modalities input to a list of uppercase modality strings."""
  if isinstance(modalities, str):
    return [m.strip().upper() for m in modalities.split(",") if m.strip()]
  return [m.strip().upper() for m in modalities if m.strip()]


def build_filter_string(
    *,
    biosample: str | None = None,
    modalities: Sequence[str] = (),
    tfs: Sequence[str] = (),
) -> str | None:
  """Builds Atlas filter string (f=) enforcing the mandatory Assay group rule."""
  filter_parts: list[str] = []

  if biosample:
    filter_parts.append(f"BIOSAMPLE_NAME:{biosample}")

  modality_map: Mapping[str, str] = {
      "RNA_SEQ": "RNA-seq",
      "DNASE": "DNase",
      "ATAC": "ATAC-seq",
      "CHIP_TF": "ChIP-TF",
      "CHIP_HISTONE": "ChIP-Histone",
      "CAGE": "CAGE",
      "PROCAP": "PRO-cap",
      "POLYADENYLATION": "Polyadenylation",
      "SPLICE_JUNCTIONS": "Splice junctions",
      "SPLICE_SITE_USAGE": "Splice site usage",
  }

  for mod in modalities:
    mod_upper = mod.upper()
    display_name = modality_map.get(mod_upper, mod)
    filter_parts.append(f"SCORER_MODALITY:{display_name}")

  for tf in tfs:
    filter_parts.append(f"ASSAY_TRANSCRIPTOR_FACTOR:{tf}")

  return ",".join(filter_parts) if filter_parts else None


def build_layout_items(
    *,
    include_avi: bool = True,
    modalities: Sequence[str] = (),
    pinned_tracks: Sequence[str] = (),
) -> str:
  """Builds Atlas layout items (lItems=)."""
  items: list[str] = []

  if include_avi:
    items.append("avi")

  for mod in modalities:
    mod_upper = mod.upper()
    items.append(f"section:{mod_upper}")

  for pinned in pinned_tracks:
    items.append(pinned)

  return ",".join(items) if items else "avi"


def _get_tf_modality_filter_str(
    *,
    biosample: str | None,
    filter_modalities: Sequence[str],
    tfs: Sequence[str],
) -> str | None:
  """Computes the filter string (f=) enforcing the mandatory Assay group rule.

  When TF filters are present, RNA_SEQ and DNASE modalities are auto-included to
  satisfy the Atlas Assay-group OR-logic requirement (see SKILL §2).

  Args:
    biosample: Optional biosample name.
    filter_modalities: Modality filters to apply.
    tfs: Transcription factor filters to apply.

  Returns:
    Formatted Atlas filter string, or None if no filters are active.
  """
  active_mods = list(filter_modalities)
  if tfs:
    for required in ("RNA_SEQ", "DNASE"):
      if required not in active_mods:
        active_mods.append(required)

  return build_filter_string(
      biosample=biosample,
      modalities=active_mods,
      tfs=tfs,
  )


def build_variant_url(config: VariantLinkConfig) -> str:
  """Builds full AlphaGenome Atlas URL for a single variant."""
  norm_var = normalize_variant_str(config.variant)
  l_items = build_layout_items(
      include_avi=config.include_avi,
      modalities=config.modalities,
      pinned_tracks=config.extra_pinned,
  )

  query: dict[str, str] = {
      "q": norm_var,
      "m": "variant",
      "lItems": l_items,
  }
  f_str = _get_tf_modality_filter_str(
      biosample=config.biosample,
      filter_modalities=config.filter_modalities,
      tfs=config.tfs,
  )
  if f_str:
    query["f"] = f_str

  return f"{BASE_ATLAS_URL}?" + urllib.parse.urlencode(
      query, quote_via=urllib.parse.quote, safe=":,-"
  )


def build_locus_url(config: LocusLinkConfig) -> str:
  """Builds full AlphaGenome Atlas URL for a genomic interval."""
  interval = config.interval.strip()
  if not interval.startswith("chr"):
    interval = f"chr{interval}"

  l_items = build_layout_items(
      include_avi=config.include_avi,
      modalities=config.modalities,
  )

  query: dict[str, str] = {
      "q": interval,
      "m": "locus",
      "lItems": l_items,
  }
  f_str = _get_tf_modality_filter_str(
      biosample=config.biosample,
      filter_modalities=config.filter_modalities,
      tfs=config.tfs,
  )
  if f_str:
    query["f"] = f_str

  return f"{BASE_ATLAS_URL}?" + urllib.parse.urlencode(
      query, quote_via=urllib.parse.quote, safe=":,-"
  )


def build_table_from_records(
    records: Sequence[Mapping[str, object]],
    *,
    biosample: str | None = None,
) -> str:
  """Formats variant records into a Markdown table with embedded Atlas links."""
  lines: list[str] = [
      (
          "| Rank | Variant | AVI Phred | Raw Score | Top Modality | Atlas"
          " Deep-Link |"
      ),
      "| :--- | :--- | :--- | :--- | :--- | :--- |",
  ]

  for idx, rec in enumerate(records, start=1):
    variant_str = str(rec.get("variant", ""))
    phred_val = rec.get("avi_phred", rec.get("phred", ""))
    raw_val = rec.get("avi_raw", rec.get("raw_score", ""))
    top_mod = rec.get("top_modality", rec.get("top_feature", "N/A"))

    phred_display = f"**{float(phred_val):.2f}**" if phred_val else "N/A"
    raw_display = f"{float(raw_val):.4f}" if raw_val else "N/A"

    if variant_str:
      url = build_variant_url(
          VariantLinkConfig(
              variant=variant_str,
              biosample=biosample,
          )
      )
      link_display = f"[View in Atlas]({url})"
    else:
      link_display = "N/A"

    lines.append(
        f"| {idx} | `{variant_str}` | {phred_display} | {raw_display} |"
        f" `{top_mod}` | {link_display} |"
    )

  return "\n".join(lines)


# ------------------------------------------------------------------------------
# Track Predictions & GENCODE Splice Junction Resolution (predict subcommand)
# ------------------------------------------------------------------------------


def _get_authenticated_atlas_client():
  """Loads .env and initializes authenticated AlphaGenome Atlas client."""
  dotenv.load_dotenv(os.path.expanduser("~/.env"))

  api_key = os.environ.get("ALPHAGENOME_API_KEY")
  if not api_key:
    print(
        "Error: ALPHAGENOME_API_KEY is not set.\n\nThe 'predict' subcommand"
        " connects to the AlphaGenome Atlas database to resolve\nexact"
        " experimental track metadata and splice junction coordinates.\n\nTo"
        " set up your API key:\n  1. Retrieve your key and add it to ~/.env:\n "
        '    echo "ALPHAGENOME_API_KEY=<your_key>" >> ~/.env\n  2. Refer to'
        " the 'credentials' skill for the safe credentials protocol,\n     or"
        " register for an API key at"
        " https://deepmind.google.com/science/alphagenome/.\n",
        file=sys.stderr,
    )
    sys.exit(1)

  return atlas.create(api_key=api_key)


GTF_URL = (
    "https://storage.googleapis.com/alphagenome/reference/gencode/"
    "hg38/gencode.v46.annotation.gtf.gz.feather"
)


def load_gtf_dataframe(custom_path: str | None = None):
  """Loads GENCODE v46 GTF annotations, downloading and caching if needed."""
  path = custom_path or os.environ.get("ALPHAGENOME_GTF_PATH")
  if path and os.path.exists(path):
    print(f"Loading GTF from {path}...")
    return pd.read_feather(path)

  cache_dir = os.path.expanduser("~/.cache/alphagenome")
  cache_path = os.path.join(cache_dir, "gencode.v46.annotation.gtf.gz.feather")
  if os.path.exists(cache_path):
    print(f"Loading GTF from cache {cache_path}...")
    return pd.read_feather(cache_path)

  target_url = path or GTF_URL
  print(f"Loading GTF from {target_url}...")
  try:
    df = pd.read_feather(target_url)
    try:
      os.makedirs(cache_dir, exist_ok=True)
      df.to_feather(cache_path)
    except Exception:
      pass
    return df
  except Exception as e:
    raise RuntimeError(
        f"Failed to load GENCODE v46 GTF annotations from {target_url}: {e}"
    ) from e


def _resolve_locus(
    *,
    gene: str | None,
    gene_id: str | None,
    interval: str | None,
    variant: genome.Variant | None,
    gtf_df,
) -> tuple[GenomicInterval, object | None]:
  """Resolves the genomic interval and target gene from inputs.

  Resolution priority: gene/gene_id > explicit interval > variant overlap.

  Args:
    gene: Gene symbol (e.g. 'BRCA1').
    gene_id: Ensembl gene ID.
    interval: Explicit interval string.
    variant: Parsed genome.Variant, or None.
    gtf_df: GENCODE v46 GTF DataFrame.

  Returns:
    Tuple of (GenomicInterval, target_gene_row or None).

  Raises:
    ValueError: If no input provides enough information to resolve a locus.
  """
  if gene or gene_id:
    return _resolve_locus_from_gene(
        gene=gene,
        gene_id=gene_id,
        interval=interval,
        gtf_df=gtf_df,
    )
  if interval:
    return parse_interval(interval), None
  if variant is not None:
    return _resolve_locus_from_variant(
        variant=variant,
        gtf_df=gtf_df,
    )
  raise ValueError("Must provide at least one of variant, gene, or interval.")


def _resolve_locus_from_gene(
    *,
    gene: str | None,
    gene_id: str | None,
    interval: str | None,
    gtf_df,
) -> tuple[GenomicInterval, object]:
  """Resolves locus from a gene symbol or Ensembl ID."""
  if gene_id:
    matches = gtf_df[gtf_df["gene_id"].str.startswith(gene_id.split(".")[0])]
  else:
    matches = gtf_df[gtf_df["gene_name"].str.upper() == gene.upper()]

  if matches.empty:
    raise ValueError(f"Gene '{gene or gene_id}' not found in GENCODE v46 GTF.")

  target_gene_row = _select_mane_preferred(matches)
  chrom = target_gene_row["Chromosome"]

  if interval:
    return parse_interval(interval), target_gene_row

  return (
      GenomicInterval(
          chrom=chrom,
          start=int(matches["Start"].min()),
          end=int(matches["End"].max()),
      ),
      target_gene_row,
  )


def _resolve_locus_from_variant(
    *,
    variant: genome.Variant,
    gtf_df,
) -> tuple[GenomicInterval, object | None]:
  """Resolves locus by overlapping variant position with gene annotations."""
  chrom = variant.chromosome
  var_pos = variant.position
  c_df = gtf_df[gtf_df["Chromosome"] == chrom]
  overlapping = c_df[(c_df["Start"] <= var_pos) & (c_df["End"] >= var_pos)]

  if not overlapping.empty:
    target_gene_row = _select_mane_preferred(overlapping)
    gene_all = c_df[c_df["gene_id"] == target_gene_row["gene_id"]]
    return (
        GenomicInterval(
            chrom=chrom,
            start=int(gene_all["Start"].min()),
            end=int(gene_all["End"].max()),
        ),
        target_gene_row,
    )

  return (
      GenomicInterval(
          chrom=chrom,
          start=max(1, var_pos - _DEFAULT_INTERGENIC_FLANK_BP),
          end=var_pos + _DEFAULT_INTERGENIC_FLANK_BP,
      ),
      None,
  )


def _collect_overlapping_genes(
    gtf_df,
    interval: GenomicInterval,
) -> tuple[list[GeneInfo], list[GeneInfo]]:
  """Finds genes overlapping an interval, split and ranked by strand."""
  c_df = gtf_df[gtf_df["Chromosome"] == interval.chrom]
  in_region = c_df[
      (c_df["Feature"] == "gene")
      & (c_df["Start"] <= interval.end)
      & (c_df["End"] >= interval.start)
  ]
  if in_region.empty:
    in_region = c_df[
        (c_df["Start"] <= interval.end) & (c_df["End"] >= interval.start)
    ]

  unique_genes = in_region.drop_duplicates(subset=["gene_id"])
  positive_genes: list[GeneInfo] = []
  negative_genes: list[GeneInfo] = []

  for _, g in unique_genes.iterrows():
    info = GeneInfo(
        gene_name=g.get("gene_name") or g["gene_id"],
        gene_id=g["gene_id"],
        strand=g["Strand"],
        is_mane="MANE_Select" in str(g.get("tag", "")),
        is_coding=g.get("gene_type") == "protein_coding",
        start=int(g["Start"]),
        end=int(g["End"]),
    )
    if info.strand == "+":
      positive_genes.append(info)
    elif info.strand == "-":
      negative_genes.append(info)

  rank_key = lambda x: (x.is_mane, x.is_coding, x.end - x.start)
  positive_genes.sort(key=rank_key, reverse=True)
  negative_genes.sort(key=rank_key, reverse=True)
  return positive_genes, negative_genes


def _pick_target_gene_id(
    target_gene_row,
    positive_genes: Sequence[GeneInfo],
    negative_genes: Sequence[GeneInfo],
) -> str | None:
  """Picks the best target gene ID from available sources."""
  if target_gene_row is not None:
    return target_gene_row["gene_id"]
  if positive_genes:
    return positive_genes[0].gene_id
  if negative_genes:
    return negative_genes[0].gene_id
  return None


def _junction_from_exon_index(gene_exons, idx: int) -> tuple[int, int]:
  """Returns junction boundaries for a variant within the exon at idx."""
  n = len(gene_exons)
  if 0 < idx < n - 1:
    return (
        int(gene_exons.iloc[idx - 1]["End"]),
        int(gene_exons.iloc[idx + 1]["Start"]),
    )
  if idx == 0:
    return int(gene_exons.iloc[0]["End"]), int(gene_exons.iloc[1]["Start"])
  return int(gene_exons.iloc[-2]["End"]), int(gene_exons.iloc[-1]["Start"])


def _find_junction_for_position(gene_exons, var_pos: int) -> tuple[int, int]:
  """Finds the splice junction flanking a variant position."""
  for i, (_, ex) in enumerate(gene_exons.iterrows()):
    if int(ex["Start"]) <= var_pos <= int(ex["End"]):
      return _junction_from_exon_index(gene_exons, i)

  for i in range(len(gene_exons) - 1):
    e1 = gene_exons.iloc[i]
    e2 = gene_exons.iloc[i + 1]
    if int(e1["End"]) < var_pos < int(e2["Start"]):
      return int(e1["End"]), int(e2["Start"])

  return int(gene_exons.iloc[0]["End"]), int(gene_exons.iloc[1]["Start"])


def _resolve_splice_junction(
    *,
    gtf_df,
    chrom: str,
    var_pos: int,
    target_gene_row,
    positive_genes: Sequence[GeneInfo],
    negative_genes: Sequence[GeneInfo],
) -> tuple[int | None, int | None]:
  """Resolves splice junction boundaries nearest to a variant position."""
  target_gid = _pick_target_gene_id(
      target_gene_row, positive_genes, negative_genes
  )
  if target_gid is None:
    return None, None

  c_df = gtf_df[gtf_df["Chromosome"] == chrom]
  target_tid = (
      target_gene_row.get("transcript_id")
      if target_gene_row is not None
      else None
  )

  if target_tid:
    gene_exons = c_df[
        (c_df["transcript_id"] == target_tid) & (c_df["Feature"] == "exon")
    ].sort_values("Start")
  else:
    gene_exons = (
        c_df[(c_df["gene_id"] == target_gid) & (c_df["Feature"] == "exon")]
        .sort_values("Start")
        .drop_duplicates(subset=["Start", "End"])
    )

  if len(gene_exons) < 2:
    return None, None

  return _find_junction_for_position(gene_exons, var_pos)


def _resolve_tracks_from_catalog(
    *,
    requested_modalities: Sequence[str],
    biosample: str,
    tf: str,
) -> dict[str, str]:
  """Resolves track names from the Atlas API scorer metadata catalog."""
  client = _get_authenticated_atlas_client()
  scorer_map = client.scorer_metadata()
  resolved: dict[str, str] = {}
  for scorer_name, s_meta in scorer_map.items():
    df_meta = s_meta.track_metadata
    if df_meta is None or df_meta.empty:
      continue
    if scorer_name.endswith("_ACTIVE"):
      continue
    if scorer_name not in requested_modalities:
      continue
    if scorer_name in resolved:
      continue
    for _, row in df_meta.iterrows():
      t_name = str(row.get("name", ""))
      b_name = str(row.get("biosample_name", ""))
      ont_id = str(row.get("ontology_curie", ""))
      if scorer_name == "CHIP_TF":
        tf_match = tf.upper() in t_name.upper()
        bio_match = (
            not biosample
            or biosample.lower() in b_name.lower()
            or biosample.lower() in t_name.lower()
        )
        if tf_match and (bio_match or "CHIP_TF" not in resolved):
          resolved["CHIP_TF"] = t_name
      else:
        if (
            biosample.lower() in b_name.lower()
            or biosample.lower() in t_name.lower()
            or biosample.lower() in ont_id.lower()
        ):
          resolved[scorer_name] = t_name
          break
  return resolved


def _build_score_tokens(
    *,
    target_variant: str,
    requested_modalities: Sequence[str],
    resolved_tracks: Mapping[str, str],
    target_gene_row,
    positive_genes: Sequence[GeneInfo],
    negative_genes: Sequence[GeneInfo],
    interval: GenomicInterval,
    junction_start: int | None,
    junction_end: int | None,
) -> tuple[list[str], list[ScoreRecord]]:
  """Builds score tokens and records for track prediction URLs."""
  tokens: list[str] = []
  records: list[ScoreRecord] = []
  gene_id = (
      _pick_target_gene_id(target_gene_row, positive_genes, negative_genes)
      or "0"
  )

  for mod in requested_modalities:
    track_name = resolved_tracks.get(mod)
    if not track_name:
      continue

    if mod == "SPLICE_JUNCTIONS":
      start = junction_start if junction_start is not None else interval.start
      end = junction_end if junction_end is not None else interval.end
      token = (
          f"{target_variant}:SPLICE_JUNCTIONS:{track_name}"
          f":Unstranded:{gene_id}:{start}:{end}"
      )
      tokens.append(token)
      records.append(
          ScoreRecord(
              modality="SPLICE_JUNCTIONS",
              track=track_name,
              strand="Gene (Junction)",
              gene=gene_id,
              score_id=token,
          )
      )

    elif mod == "RNA_SEQ":
      if positive_genes:
        g = positive_genes[0]
        token = (
            f"{target_variant}:RNA_SEQ:{track_name}:Unstranded:{g.gene_id}:0:0"
        )
        tokens.append(token)
        records.append(
            ScoreRecord(
                modality="RNA_SEQ (+ strand)",
                track=track_name,
                strand="+ (Forward)",
                gene=f"{g.gene_name} ({g.gene_id})",
                score_id=token,
            )
        )
      if negative_genes:
        g = negative_genes[0]
        token = (
            f"{target_variant}:RNA_SEQ:{track_name}:Unstranded:{g.gene_id}:0:0"
        )
        tokens.append(token)
        records.append(
            ScoreRecord(
                modality="RNA_SEQ (- strand)",
                track=track_name,
                strand="- (Reverse)",
                gene=f"{g.gene_name} ({g.gene_id})",
                score_id=token,
            )
        )

    elif mod in (
        "DNASE",
        "ATAC",
        "CHIP_TF",
        "CHIP_HISTONE",
        "CAGE",
        "PROCAP",
    ):
      token = f"{target_variant}:{mod}:{track_name}:Unstranded:None:None:None"
      tokens.append(token)
      records.append(
          ScoreRecord(
              modality=mod,
              track=track_name,
              strand="Unstranded",
              gene="None",
              score_id=token,
          )
      )

  return tokens, records


def generate_track_prediction_link(
    *,
    variant: str | None = None,
    gene: str | None = None,
    gene_id: str | None = None,
    interval: str | None = None,
    biosample: str = "Muscle_Skeletal",
    modalities: Sequence[str] | str = (
        "SPLICE_JUNCTIONS",
        "RNA_SEQ",
        "DNASE",
        "CHIP_TF",
    ),
    tf: str = "CTCF",
    tp_renames: str | None = None,
    tp_legend_title: str | None = None,
    gtf_path: str | None = None,
) -> dict[str, object]:
  """Constructs Atlas track prediction deep-links and resolves locus structures.

  Args:
    variant: Variant in chr:pos:ref>alt format.
    gene: Target gene symbol.
    gene_id: Ensembl gene ID.
    interval: Explicit genomic interval viewport (chr:start-end).
    biosample: Target biosample or tissue name.
    modalities: Modalities to include (comma-separated string or sequence).
    tf: Transcription factor name for ChIP-TF tracks.
    tp_renames: Custom track rename overrides for the URL.
    tp_legend_title: Custom legend title for the track prediction card.
    gtf_path: Path or URL to the GTF feather file.

  Returns:
    Dict with 'url', 'interval', 'variant', 'target_gene',
    'positive_strand_genes', 'negative_strand_genes', and 'scores'.
  """
  gtf_df = load_gtf_dataframe(gtf_path)
  requested_modalities = _parse_modality_list(modalities)
  parsed_var = parse_variant_components(variant) if variant else None

  locus, target_gene_row = _resolve_locus(
      gene=gene,
      gene_id=gene_id,
      interval=interval,
      variant=parsed_var,
      gtf_df=gtf_df,
  )

  positive_genes, negative_genes = _collect_overlapping_genes(gtf_df, locus)

  junction_start, junction_end = None, None
  if "SPLICE_JUNCTIONS" in requested_modalities and parsed_var is not None:
    junction_start, junction_end = _resolve_splice_junction(
        gtf_df=gtf_df,
        chrom=locus.chrom,
        var_pos=parsed_var.position,
        target_gene_row=target_gene_row,
        positive_genes=positive_genes,
        negative_genes=negative_genes,
    )

  resolved_tracks = _resolve_tracks_from_catalog(
      requested_modalities=requested_modalities,
      biosample=biosample,
      tf=tf,
  )

  target_var = variant or (
      f"{locus.chrom}:{locus.start + (locus.end - locus.start) // 2}:A>G"
  )
  score_tokens, score_records = _build_score_tokens(
      target_variant=target_var,
      requested_modalities=requested_modalities,
      resolved_tracks=resolved_tracks,
      target_gene_row=target_gene_row,
      positive_genes=positive_genes,
      negative_genes=negative_genes,
      interval=locus,
      junction_start=junction_start,
      junction_end=junction_end,
  )

  l_items_str = build_layout_items(
      include_avi=True,
      modalities=requested_modalities,
  )

  query_dict: dict[str, str] = {
      "q": variant or str(locus),
      "m": "variant" if variant else "locus",
      "lItems": l_items_str,
      "scores": ",".join(score_tokens),
      "i": str(locus),
  }
  if tp_renames:
    query_dict["tpRenames"] = tp_renames
  if tp_legend_title:
    query_dict["tpLegendTitle"] = tp_legend_title

  final_url = f"{BASE_ATLAS_URL}/track-predictions?" + urllib.parse.urlencode(
      query_dict, quote_via=urllib.parse.quote, safe=":,-"
  )

  return {
      "url": final_url,
      "interval": str(locus),
      "variant": variant,
      "target_gene": (
          target_gene_row["gene_name"] if target_gene_row is not None else None
      ),
      "positive_strand_genes": [dataclasses.asdict(g) for g in positive_genes],
      "negative_strand_genes": [dataclasses.asdict(g) for g in negative_genes],
      "scores": [dataclasses.asdict(r) for r in score_records],
  }


def main() -> None:
  parser = argparse.ArgumentParser(
      description=(
          "AlphaGenome Atlas Deep-Link & Track Predictions Generator CLI"
      ),
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  subparsers = parser.add_subparsers(dest="subcommand", required=True)

  # Subcommand: variant
  p_var = subparsers.add_parser(
      "variant",
      help="Generate an AlphaGenome Atlas exploration URL for a single variant",
  )
  p_var.add_argument("variant", help="Variant in chr:pos:ref>alt format")
  p_var.add_argument(
      "--biosample",
      "-b",
      help="Biosample or cell line name (e.g. K562, HepG2, Muscle_Skeletal)",
  )
  p_var.add_argument(
      "--modalities",
      "-m",
      default="RNA_SEQ,DNASE,CHIP_TF",
      help="Comma-separated modalities to include in layout",
  )
  p_var.add_argument(
      "--tf",
      dest="tfs",
      action="append",
      default=[],
      help="Transcription factor names for ChIP-TF tracks (e.g. GATA1, CTCF)",
  )
  p_var.add_argument(
      "--no_avi",
      action="store_true",
      help="Exclude top-level AVI score track",
  )

  # Subcommand: locus
  p_loc = subparsers.add_parser(
      "locus",
      help=(
          "Generate an AlphaGenome Atlas locus exploration URL for an interval"
      ),
  )
  p_loc.add_argument(
      "interval",
      help="Genomic coordinate interval (e.g. chr11:5288500-5290500)",
  )
  p_loc.add_argument(
      "--biosample",
      "-b",
      help="Biosample or cell line filter",
  )
  p_loc.add_argument(
      "--modalities",
      "-m",
      default="RNA_SEQ,DNASE,CHIP_TF",
      help="Comma-separated modalities to include",
  )
  p_loc.add_argument(
      "--tf",
      dest="tfs",
      action="append",
      default=[],
      help="Transcription factors for ChIP-TF filters",
  )
  p_loc.add_argument(
      "--no_avi",
      action="store_true",
      help="Exclude top-level AVI score track",
  )

  # Subcommand: table
  p_tab = subparsers.add_parser(
      "table",
      help=(
          "Format variant records or a JSON file into a Markdown table with"
          " embedded Atlas links"
      ),
  )
  p_tab.add_argument(
      "--input",
      "-i",
      help="Path to JSON file containing variant records",
  )
  p_tab.add_argument(
      "--variant",
      "-v",
      action="append",
      default=[],
      help="Variant string(s) to include in table",
  )
  p_tab.add_argument(
      "--biosample",
      "-b",
      help="Optional biosample filter to attach to links",
  )

  # Subcommand: track-predictions (comparison URLs)
  p_pred = subparsers.add_parser(
      "track-predictions",
      help=(
          "Construct /atlas/track-predictions comparison chart URLs with paired"
          " RNA-seq + splicing"
      ),
  )
  p_pred.add_argument(
      "--variant",
      "-v",
      help="Variant string in chr:pos:ref>alt format",
  )
  p_pred.add_argument(
      "--gene",
      "-g",
      help="Target gene symbol (e.g. CAPN3)",
  )
  p_pred.add_argument(
      "--gene_id",
      help="Target Ensembl gene ID (e.g. ENSG00000092529.26)",
  )
  p_pred.add_argument(
      "--interval",
      "--region",
      "-i",
      dest="interval",
      help="Genomic interval viewport in chr:start-end format",
  )
  p_pred.add_argument(
      "--biosample",
      "-b",
      default="Muscle_Skeletal",
      help=(
          "Target biosample or tissue query (e.g. Muscle_Skeletal, K562,"
          " Whole_Blood)"
      ),
  )
  p_pred.add_argument(
      "--modalities",
      "-m",
      default="SPLICE_JUNCTIONS,RNA_SEQ,DNASE,CHIP_TF",
      help=(
          "Comma-separated modalities to include (SPLICE_JUNCTIONS, RNA_SEQ,"
          " DNASE, ATAC, CHIP_TF)"
      ),
  )
  p_pred.add_argument(
      "--tf",
      default="CTCF",
      help="Transcription factor name for ChIP-TF tracks (e.g. CTCF, GATA1)",
  )
  p_pred.add_argument(
      "--rename",
      "--tp_renames",
      dest="tp_renames",
      help="Custom track rename overrides (e.g. ScoreId:CustomTitle)",
  )
  p_pred.add_argument(
      "--legend_title",
      "--tp_legend_title",
      dest="tp_legend_title",
      help="Custom legend title for the track prediction card",
  )
  p_pred.add_argument(
      "--gtf_path",
      help=(
          "Path or URL to the GTF feather file (defaults to"
          " $ALPHAGENOME_GTF_PATH or the public GCS URL)"
      ),
  )
  p_pred.add_argument(
      "--format",
      choices=["table", "json", "url"],
      default="table",
      help="Output format",
  )

  args = parser.parse_args()

  def _parse_modalities(raw: str) -> list[str]:
    return [m.strip() for m in raw.split(",") if m.strip()]

  if args.subcommand == "variant":
    modalities = _parse_modalities(args.modalities)
    config = VariantLinkConfig(
        variant=args.variant,
        biosample=args.biosample,
        modalities=modalities,
        tfs=args.tfs,
        include_avi=not args.no_avi,
    )
    url = build_variant_url(config)
    print(url)

  elif args.subcommand == "locus":
    modalities = _parse_modalities(args.modalities)
    config = LocusLinkConfig(
        interval=args.interval,
        biosample=args.biosample,
        modalities=modalities,
        tfs=args.tfs,
        include_avi=not args.no_avi,
    )
    url = build_locus_url(config)
    print(url)

  elif args.subcommand == "table":
    records: list[dict[str, object]] = []
    if args.input:
      with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
          records.extend(data)
        elif isinstance(data, dict):
          records.append(data)
    for v in args.variant:
      records.append({"variant": v})

    if not records:
      print(
          "Error: No variants provided via --input or --variant.",
          file=sys.stderr,
      )
      sys.exit(1)

    table_md = build_table_from_records(
        records,
        biosample=args.biosample,
    )
    print(table_md)

  elif args.subcommand == "track-predictions":
    try:
      result = generate_track_prediction_link(
          variant=args.variant,
          gene=args.gene,
          gene_id=args.gene_id,
          interval=args.interval,
          biosample=args.biosample,
          modalities=args.modalities,
          tf=args.tf,
          tp_renames=args.tp_renames,
          tp_legend_title=args.tp_legend_title,
          gtf_path=args.gtf_path,
      )
    except Exception as e:
      print(f"Error: {e}", file=sys.stderr)
      sys.exit(1)

    if args.format == "url":
      print(result["url"])
      return

    if args.format == "json":
      print(json.dumps(result, indent=2))
      return

    print("\n### AlphaGenome Atlas Track Predictions Deep-Link")
    print(f"\n**Generated URL**:\n{result['url']}\n")
    print(f"**Genomic Interval Viewport (`i`)**: `{result['interval']}`")
    if result["variant"]:
      print(f"**Target Variant (`q`)**: `{result['variant']}`")
    if result["target_gene"]:
      print(f"**Target Gene**: `{result['target_gene']}`")
    print()

    positive_genes = result["positive_strand_genes"]
    negative_genes = result["negative_strand_genes"]
    chrom = str(result["interval"]).split(":", maxsplit=1)[0]

    print("#### Overlapping Genes in Interval:")
    print(f"- **(+) Forward Strand Genes ({len(positive_genes)})**:")
    for g in positive_genes[:5]:
      mane_tag = " [MANE Select]" if g["is_mane"] else ""
      print(
          f"  - {g['gene_name']} ({g['gene_id']}){mane_tag}:"
          f" `{chrom}:{g['start']}-{g['end']}`"
      )
    if len(positive_genes) > 5:
      print(f"  - ... and {len(positive_genes) - 5} more (+)-strand genes")

    print(f"- **(-) Reverse Strand Genes ({len(negative_genes)})**:")
    for g in negative_genes[:5]:
      mane_tag = " [MANE Select]" if g["is_mane"] else ""
      print(
          f"  - {g['gene_name']} ({g['gene_id']}){mane_tag}:"
          f" `{chrom}:{g['start']}-{g['end']}`"
      )
    if len(negative_genes) > 5:
      print(f"  - ... and {len(negative_genes) - 5} more (-)-strand genes")
    print()

    print("#### Composed Track Prediction Scores (`scores`):")
    print("| Modality | Strand | Track Name | Gene ID / Target |")
    print("| :--- | :--- | :--- | :--- |")
    for r in result["scores"]:
      print(
          f"| `{r['modality']}` | {r['strand']} | `{r['track']}` |"
          f" {r['gene']} |"
      )
    print()


if __name__ == "__main__":
  main()
