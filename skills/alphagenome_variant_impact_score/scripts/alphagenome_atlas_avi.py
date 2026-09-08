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

"""AlphaGenome Variant Impact (AVI) CLI.

Provides unified subcommands:
1. `annotate`: Annotate input VCF/tabular variant files in Ensembl VEP CSQ
format and rank hits.
2. `query`: Query one or more variant strings to get scores, full 18-modality
feature importances, and underlying track attributions.
3. `region`: Perform in silico saturation mutagenesis across a genomic window
and rank variants.
5. `gtf`: Query GENCODE v46 gene annotations, extract exons, CDS, UTRs, and
compute canonical and exon-skipping splice junction coordinates to build exact
Atlas track prediction `ScoreId` tokens.

Powered by the AlphaGenome Atlas API (`alphagenome.atlas.atlas`) and
AlphaGenome GENCODE v46 annotations.

Usage:
  uv run alphagenome_atlas_avi.py query chr9:128225994:G>A
  uv run alphagenome_atlas_avi.py query chr9:128225994:G>A --include_track_info
  uv run alphagenome_atlas_avi.py annotate -i variants.vcf -o annotated.vcf
  uv run alphagenome_atlas_avi.py region -r chr9:128225900-128226050
  uv run alphagenome_atlas_avi.py metadata --scorers
  uv run alphagenome_atlas_avi.py metadata --tracks --query "GATA1"
  uv run alphagenome_atlas_avi.py metadata --features
  uv run alphagenome_atlas_avi.py gtf --gene CAPN3 --exons
  uv run alphagenome_atlas_avi.py gtf --variant chr15:42387805:C>G --junctions
"""

import argparse
from collections.abc import Mapping, MutableSequence, Sequence
import dataclasses
import enum
import json
import math
import os
import re
import sys

from alphagenome.atlas import atlas
from alphagenome.data import genome
import anndata
import dotenv
import numpy as np
import pandas as pd
import polars as pl
import pysam
import tabulate

_ATLAS_TRACK_PREDICTIONS_URL = (
    'https://deepmind.google.com/science/alphagenome/atlas/track-predictions'
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class TrackAttribution:
  """Detailed provenance for an assay track driving a feature maximum.

  Attributes:
    scorer: Name or identifier of the scorer modality.
    track_index: Index of the assay track.
    track_name: Name or description of the track.
    biosample_name: Name of the biosample, cell line, or tissue.
    ontology_curie: Ontology CURIE for the biosample (e.g. UBERON or CL term).
    raw_score: Raw attribution score or effect size for this track.
    gene_name: Optional HGNC gene symbol associated with the track.
    gene_id: Optional Ensembl gene ID associated with the track.
    details: Summary string describing the maximum signal or attribution
      details.
  """

  scorer: str
  track_index: int
  track_name: str
  biosample_name: str
  ontology_curie: str
  raw_score: float
  gene_name: str | None = None
  gene_id: str | None = None
  details: str = ''


@dataclasses.dataclass
class VariantRecord(genome.Variant):
  """Genomic variant annotated with AVI scores and feature importances.

  Subclasses `genome.Variant` to inherit chromosome, position,
  reference_bases, alternate_bases, validation, and serialization.
  """

  raw_score: float | None = dataclasses.field(
      default=None, compare=False, hash=False
  )
  phred_score: float | None = dataclasses.field(
      default=None, compare=False, hash=False
  )
  quantile: float | None = dataclasses.field(
      default=None, compare=False, hash=False
  )
  top_feature_name: str | None = dataclasses.field(
      default=None, compare=False, hash=False
  )
  top_feature_value: float | None = dataclasses.field(
      default=None, compare=False, hash=False
  )
  feature_importances: dict[str, float] = dataclasses.field(
      default_factory=dict, compare=False, hash=False
  )
  track_attributions: dict[str, TrackAttribution] = dataclasses.field(
      default_factory=dict, compare=False, hash=False
  )
  raw_record: pysam.VariantRecord | None = dataclasses.field(
      default=None, repr=False, compare=False, hash=False
  )

  @property
  def variant_id(self) -> str:
    """Returns formatted variant string: chr:pos:ref>alt."""
    return str(self)

  @property
  def top_percentile(self) -> float | None:
    """Returns the exact top percentile (e.g. 0.10715 for top 0.11% impact)."""
    if self.quantile is not None:
      return self.quantile * 100.0
    if self.phred_score is not None:
      return (10.0 ** (-self.phred_score / 10.0)) * 100.0
    return None

  def to_row(
      self,
      *,
      include_features: bool = False,
      include_track_info: bool = False,
      rank: int | None = None,
  ) -> dict[str, object]:
    """Serializes to a flat dictionary for tabular/JSON export."""
    row: dict[str, object] = {}
    if rank is not None:
      row['rank'] = rank
    row.update({
        'variant': self.variant_id,
        'chromosome': self.chromosome,
        'position': self.position,
        'ref': self.reference_bases,
        'alt': self.alternate_bases,
        'avi_phred': self.phred_score,
        'avi_raw': self.raw_score,
        'avi_quantile': self.quantile,
        'top_percentile': self.top_percentile,
        'top_modality': self.top_feature_name,
        'top_feature_importance': self.top_feature_value,
    })
    if include_features:
      for feat_name, feat_val in self.feature_importances.items():
        row[f'fi_{feat_name}'] = feat_val
    if include_track_info and self.track_attributions:
      for feat_name, attr in self.track_attributions.items():
        row[f'track_idx_{feat_name}'] = int(attr.track_index)
        row[f'track_name_{feat_name}'] = str(attr.track_name)
        row[f'track_biosample_{feat_name}'] = str(attr.biosample_name)
        if attr.gene_name:
          row[f'track_gene_{feat_name}'] = str(attr.gene_name)
    return row


class AviFeature(enum.Enum):
  """AlphaGenome Variant Impact (AVI) biological modality features."""

  MERGED_SPLICING = (
      'Splicing',
      'Splicing',
      ('SPLICE_SITES', 'SPLICE_SITE_USAGE', 'SPLICE_JUNCTIONS'),
  )
  MAX_ABS_RNA_SEQ = ('RNA-seq', 'Transcription / RNA-seq', ('RNA_SEQ',))
  MAX_ABS_ATAC = ('ATAC-seq', 'Chromatin Accessibility', ('ATAC',))
  MAX_ABS_DNASE = ('DNASE-seq', 'Chromatin Accessibility', ('DNASE',))
  MAX_ABS_CHIP_TF = (
      'ChIP-TF',
      'Transcription Factor Binding',
      ('CHIP_TF',),
  )
  MAX_ABS_CHIP_HISTONE = (
      'ChIP-Histone',
      'Histone Modification',
      ('CHIP_HISTONE',),
  )
  MAX_ABS_CAGE = ('CAGE', 'Transcription / RNA-seq', ('CAGE',))
  MAX_ABS_PROCAP = ('PRO-cap', 'Transcription / RNA-seq', ('PROCAP',))
  MAX_ABS_POLYADENYLATION = (
      'Polyadenylation',
      'Transcription / RNA-seq',
      ('POLYADENYLATION',),
  )
  MAX_ABS_CONTACT_MAPS = (
      '3D Genome Contacts',
      '3D Genome Organization',
      ('CONTACT_MAPS',),
  )
  ALPHAMISSENSE = ('AlphaMissense', 'Protein Impact', ())
  CACTUS_241_WAY = ('Cactus', 'Evolutionary Conservation', ())
  PROTEIN_TERMINATION = ('Protein Termination', 'Coding Consequence', ())
  START_LOST = ('Start Lost', 'Coding Consequence', ())
  STOP_LOST = ('Stop Lost', 'Coding Consequence', ())
  PHASTCONS_470_WAY = ('PhastCons 470', 'Evolutionary Conservation', ())
  IS_INSERTION = ('Insertion', 'Structural Variant', ())
  IS_DELETION = ('Deletion', 'Structural Variant', ())

  display_name: str
  category: str
  scorers: Sequence[str]

  def __init__(
      self,
      display_name: str,
      category: str,
      scorers: Sequence[str] = (),
  ):
    self.display_name = display_name
    self.category = category
    self.scorers = scorers

  @property
  def is_sequence_or_conservation(self) -> bool:
    """Returns True if the feature has no associated experimental scorers."""
    return not self.scorers

  @classmethod
  def from_key(cls, key: str) -> 'AviFeature | None':
    """Returns the AviFeature corresponding to the given string key or None."""
    try:
      return cls[key]
    except KeyError:
      return None


_REQUESTED_SCORERS = ('AVI_SCORE', 'AVI_SCORE_FEATURE_IMPORTANCE')

_SPLICE_SUB_SCORERS: Sequence[tuple[str, float, str]] = (
    ('SPLICE_SITES', 1.0, 'Core Splice Site'),
    ('SPLICE_SITE_USAGE', 1.0, 'Splice Usage'),
    ('SPLICE_JUNCTIONS', 1.0 / 5.0, 'Novel Junction'),
)


def format_feature_name(name: str | None) -> str:
  """Maps backend feature name to website display name if available."""
  if not name:
    return ''
  feat = AviFeature.from_key(name)
  return feat.display_name if feat is not None else name


def _normalize_chrom(chromosome: str, *, with_chr_prefix: bool = True) -> str:
  """Normalizes chromosome names (e.g. 'chr1' <-> '1')."""
  chrom_str = str(chromosome).strip()
  has_chr = chrom_str.lower().startswith('chr')
  if with_chr_prefix:
    return chrom_str if has_chr else f'chr{chrom_str}'
  return chrom_str[3:] if has_chr else chrom_str


def _cdf_to_phred(cdf_quantile: float) -> tuple[float, float]:
  """Converts a CDF quantile to (tail_quantile, phred_score).

  Args:
    cdf_quantile: The cumulative distribution function quantile value.

  Returns:
    Tuple of (tail_quantile, phred_score) where phred = -10 * log10(tail).
  """
  tail = max(1e-7, 1.0 - cdf_quantile)
  return tail, -10.0 * math.log10(tail)


def _resolve_feature_names(adata: anndata.AnnData) -> Sequence[str]:
  """Extracts ordered feature names from an AnnData var axis."""
  if (
      hasattr(adata, 'var')
      and adata.var is not None
      and 'name' in adata.var.columns
  ):
    return list(adata.var['name'])
  if hasattr(adata, 'var_names'):
    return list(adata.var_names)
  return ()


def _compute_feature_attributions(
    feature_values: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[dict[str, float], str | None, float | None]:
  """Builds feature importance dict and identifies the top contributor.

  Args:
    feature_values: 1-D array of attribution weights per feature.
    feature_names: Ordered feature names matching the values.

  Returns:
    Tuple of (importances_dict, top_feature_display_name, top_feature_value).
  """
  if len(feature_names) == 0 or len(feature_names) != len(feature_values):
    return {}, None, None

  importances: dict[str, float] = {}
  for name, value in zip(feature_names, feature_values, strict=True):
    importances[str(name)] = float(value)

  max_index = int(np.argmax(np.abs(feature_values)))
  top_name = format_feature_name(str(feature_names[max_index]))
  top_value = float(feature_values[max_index])
  return importances, top_name, top_value


@dataclasses.dataclass(frozen=True, kw_only=True)
class _AnnDataMaxHit:
  """Result of finding the maximum absolute value in an AnnData matrix."""

  gene_idx: int
  track_idx: int
  raw_value: float
  track_name: str
  biosample_name: str
  ontology_curie: str
  gene_name: str | None
  gene_id: str | None


def _find_adata_max(
    adata: anndata.AnnData,
    *,
    use_abs: bool = False,
    scale: float = 1.0,
) -> _AnnDataMaxHit | None:
  """Finds the element with maximum (optionally absolute) value in an AnnData.

  Args:
    adata: AnnData object with a 2-D X matrix (genes × tracks).
    use_abs: If True, find argmax of |X| but return the original value.
    scale: Multiplicative factor applied to the raw value (e.g. 1/5 for splice
      junctions).

  Returns:
    An _AnnDataMaxHit or None if the matrix is empty.
  """
  if adata.X is None or adata.X.size == 0:
    return None

  arr = np.nan_to_num(adata.X)
  search_arr = np.abs(arr) if use_abs else arr
  flat_idx = int(np.argmax(search_arr))
  g_idx, t_idx = (int(x) for x in np.unravel_index(flat_idx, arr.shape))
  raw_val = float(arr[g_idx, t_idx]) * scale

  var_df = adata.var
  has_var = var_df is not None and len(var_df) > t_idx
  t_name = str(var_df.iloc[t_idx].get('name', '')) if has_var else ''
  b_name = str(var_df.iloc[t_idx].get('biosample_name', '')) if has_var else ''
  onto = str(var_df.iloc[t_idx].get('ontology_curie', '')) if has_var else ''

  g_name: str | None = None
  g_id: str | None = None
  if len(adata.obs) > g_idx:
    obs_row = adata.obs.iloc[g_idx]
    g_name = str(obs_row.get('gene_name', obs_row.get('name', ''))) or None
    g_id = str(obs_row.get('gene_id', '')) or None

  return _AnnDataMaxHit(
      gene_idx=g_idx,
      track_idx=t_idx,
      raw_value=raw_val,
      track_name=t_name,
      biosample_name=b_name,
      ontology_curie=onto,
      gene_name=g_name,
      gene_id=g_id,
  )


def _resolve_track_attributions(
    variant_obj: genome.Variant,
    client: atlas.AtlasClient,
    feature_importances: Mapping[str, float],
) -> dict[str, TrackAttribution]:
  """Queries underlying assay scorers to resolve winning track indices."""
  needed_scorers: set[str] = set()
  for feat_name in feature_importances:
    feat = AviFeature.from_key(feat_name)
    if feat is not None and feat.scorers:
      needed_scorers.update(feat.scorers)

  if not needed_scorers:
    return {}

  assay_scores = client.query_variant(
      variant_obj, requested_scorers=list(needed_scorers)
  )
  attributions: dict[str, TrackAttribution] = {}

  for feat_name, feat_weight in feature_importances.items():
    feat = AviFeature.from_key(feat_name)
    if feat is AviFeature.MERGED_SPLICING:
      # Splicing: SPLICE_SITES + SPLICE_SITE_USAGE + (SPLICE_JUNCTIONS / 5)
      sub_attrs: MutableSequence[tuple[float, TrackAttribution]] = []

      for scorer_key, scale, label in _SPLICE_SUB_SCORERS:
        if scorer_key not in assay_scores:
          continue
        hit = _find_adata_max(assay_scores[scorer_key], scale=scale)
        if hit is None:
          continue
        # SPLICE_SITES has no per-track biosample; use 'Consensus'.
        b_name = hit.biosample_name or (
            'Consensus' if scorer_key == 'SPLICE_SITES' else ''
        )
        detail_ctx = hit.track_name if scorer_key == 'SPLICE_SITES' else b_name
        sub_attrs.append((
            hit.raw_value,
            TrackAttribution(
                scorer=scorer_key,
                track_index=hit.track_idx,
                track_name=hit.track_name or scorer_key.lower(),
                biosample_name=b_name,
                ontology_curie=hit.ontology_curie or 'N/A',
                raw_score=hit.raw_value,
                gene_name=hit.gene_name,
                details=f'{label} ({detail_ctx}): {hit.raw_value:.3f}',
            ),
        ))

      if sub_attrs:
        sub_attrs.sort(key=lambda x: x[0], reverse=True)
        dominant_attr = sub_attrs[0][1]
        all_details = ' | '.join(a[1].details for a in sub_attrs)
        attributions['MERGED_SPLICING'] = dataclasses.replace(
            dominant_attr, details=all_details
        )

    elif feat is not None and feat.scorers:
      scorer_name = feat.scorers[0]
      if scorer_name in assay_scores:
        hit = _find_adata_max(assay_scores[scorer_name], use_abs=True)
        if hit is not None:
          attributions[feat_name] = TrackAttribution(
              scorer=scorer_name,
              track_index=hit.track_idx,
              track_name=hit.track_name,
              biosample_name=hit.biosample_name,
              ontology_curie=hit.ontology_curie,
              raw_score=hit.raw_value,
              gene_name=hit.gene_name,
              gene_id=hit.gene_id,
              details=f'Max Abs: {hit.raw_value:+.4f}',
          )
    else:
      # Sequence / Conservation feature
      attributions[feat_name] = TrackAttribution(
          scorer='SEQUENCE_OR_CONSERVATION',
          track_index=0,
          track_name=format_feature_name(feat_name),
          biosample_name='N/A',
          ontology_curie='N/A',
          raw_score=feat_weight,
          details='Sequence annotation',
      )

  return attributions


def _variant_sort_key(record: VariantRecord) -> float:
  """Sort key for ranking variants by Phred score descending, then raw."""
  if record.phred_score is not None and not np.isnan(record.phred_score):
    return record.phred_score
  if record.raw_score is not None and not np.isnan(record.raw_score):
    return record.raw_score
  return -999.0


def _rank_and_filter(
    scored: Sequence[VariantRecord],
    *,
    min_phred: float = 0.0,
    top_k: int | None = None,
) -> tuple[Sequence[VariantRecord], Sequence[VariantRecord]]:
  """Sorts by score descending and filters to top candidates.

  Args:
    scored: Scored variant records.
    min_phred: Minimum Phred score threshold for inclusion.
    top_k: Maximum number of top hits to return.

  Returns:
    Tuple of (all_ranked, top_hits).
  """
  ranked = sorted(scored, key=_variant_sort_key, reverse=True)
  top_hits = [
      v
      for v in ranked
      if (v.phred_score is not None and v.phred_score >= min_phred)
      or (v.phred_score is None and v.raw_score is not None)
  ]
  if top_k is not None:
    top_hits = top_hits[:top_k]
  return ranked, top_hits


def _to_variant_record(variant_obj: genome.Variant | str) -> VariantRecord:
  """Coerces a genome.Variant or string to a VariantRecord."""
  if isinstance(variant_obj, VariantRecord):
    return variant_obj
  if isinstance(variant_obj, genome.Variant):
    return VariantRecord(
        chromosome=variant_obj.chromosome,
        position=variant_obj.position,
        reference_bases=variant_obj.reference_bases,
        alternate_bases=variant_obj.alternate_bases,
    )
  parsed = genome.Variant.from_str(str(variant_obj))
  return VariantRecord(
      chromosome=parsed.chromosome,
      position=parsed.position,
      reference_bases=parsed.reference_bases,
      alternate_bases=parsed.alternate_bases,
  )


def get_atlas_client() -> atlas.AtlasClient:
  """Initializes the AlphaGenome Atlas client using ALPHAGENOME_API_KEY."""
  api_key = os.environ.get('ALPHAGENOME_API_KEY')
  if not api_key:
    raise ValueError(
        'ALPHAGENOME_API_KEY environment variable not set. Please ensure'
        ' ALPHAGENOME_API_KEY is configured in ~/.env.'
    )
  return atlas.create(api_key)


def _score_single_variant(
    record: VariantRecord,
    client: atlas.AtlasClient,
    *,
    include_track_info: bool = False,
) -> VariantRecord:
  """Scores a single VariantRecord via the Atlas API."""
  normalized_chrom = _normalize_chrom(record.chromosome, with_chr_prefix=True)
  variant_obj = genome.Variant(
      chromosome=normalized_chrom,
      position=record.position,
      reference_bases=record.reference_bases,
      alternate_bases=record.alternate_bases,
  )
  scores = client.query_variant(
      variant_obj, requested_scorers=_REQUESTED_SCORERS
  )

  raw_score: float | None = None
  quantile_val: float | None = None
  phred_val: float | None = None
  feature_importances: dict[str, float] = {}
  top_feature_name: str | None = None
  top_feature_value: float | None = None

  if 'AVI_SCORE' in scores:
    avi_adata = scores['AVI_SCORE']
    if avi_adata.X is not None and avi_adata.X.size > 0:
      raw_score = float(np.ravel(avi_adata.X)[0])
    if 'quantiles' in avi_adata.layers:
      cdf_quantile = float(np.ravel(avi_adata.layers['quantiles'])[0])
      quantile_val, phred_val = _cdf_to_phred(cdf_quantile)

  if 'AVI_SCORE_FEATURE_IMPORTANCE' in scores:
    fi_adata = scores['AVI_SCORE_FEATURE_IMPORTANCE']
    if fi_adata.X is not None and fi_adata.X.size > 0:
      feature_values = np.ravel(fi_adata.X)
      feature_names = _resolve_feature_names(fi_adata)
      feature_importances, top_feature_name, top_feature_value = (
          _compute_feature_attributions(feature_values, feature_names)
      )

  track_attributions: dict[str, TrackAttribution] = {}
  if include_track_info and feature_importances:
    track_attributions = _resolve_track_attributions(
        variant_obj, client, feature_importances
    )

  return dataclasses.replace(
      record,
      raw_score=raw_score,
      phred_score=phred_val,
      quantile=quantile_val,
      top_feature_name=top_feature_name,
      top_feature_value=top_feature_value,
      feature_importances=feature_importances,
      track_attributions=track_attributions,
  )


def score_variant_batch(
    variants: Sequence[VariantRecord],
    client: atlas.AtlasClient,
    *,
    include_track_info: bool = False,
) -> Sequence[VariantRecord]:
  """Scores a sequence of VariantRecord objects using the Atlas client."""
  return [
      _score_single_variant(
          record, client, include_track_info=include_track_info
      )
      for record in variants
  ]


def _read_variants_vcf(
    path: str,
) -> tuple[Sequence[VariantRecord], pysam.VariantHeader]:
  """Parses variants from a VCF / VCF.GZ file."""
  records: MutableSequence[VariantRecord] = []
  with pysam.VariantFile(path) as vcf_in:
    for record in vcf_in.fetch():
      for alt in record.alts or ():
        records.append(
            VariantRecord(
                chromosome=record.chrom,
                position=int(record.pos),
                reference_bases=record.ref,
                alternate_bases=alt,
                raw_record=record,
            )
        )
    header = vcf_in.header.copy()
  return records, header


def _read_variants_tabular(path: str) -> Sequence[VariantRecord]:
  """Parses variants from CSV, TSV, or Parquet."""
  if path.endswith('.parquet'):
    table_df = pl.read_parquet(path)
  elif path.endswith('.tsv') or path.endswith('.txt'):
    table_df = pl.read_csv(path, separator='\t')
  else:
    table_df = pl.read_csv(path)

  column_lookup = {col.lower(): col for col in table_df.columns}
  chrom_col = (
      column_lookup.get('chrom')
      or column_lookup.get('chromosome')
      or column_lookup.get('#chrom')
      or column_lookup.get('#chromosome')
  )
  pos_col = (
      column_lookup.get('pos')
      or column_lookup.get('position')
      or column_lookup.get('start')
  )
  ref_col = (
      column_lookup.get('ref')
      or column_lookup.get('reference')
      or column_lookup.get('reference_bases')
  )
  alt_col = (
      column_lookup.get('alt')
      or column_lookup.get('alternate')
      or column_lookup.get('alternate_bases')
  )

  if not (chrom_col and pos_col and ref_col and alt_col):
    raise ValueError(
        'Input tabular file must contain chromosome, position, ref, and alt'
        f' columns. Found columns: {table_df.columns}'
    )

  records: MutableSequence[VariantRecord] = []
  for row in table_df.iter_rows(named=True):
    records.append(
        VariantRecord(
            chromosome=str(row[chrom_col]),
            position=int(row[pos_col]),
            reference_bases=str(row[ref_col]),
            alternate_bases=str(row[alt_col]),
        )
    )
  return records


def _write_tabular(
    output_path: str,
    rows: Sequence[dict[str, object]],
) -> None:
  """Writes rows to CSV, TSV, Parquet, or JSON based on file extension."""
  if output_path.endswith('.json'):
    with open(output_path, 'w', encoding='utf-8') as json_file:
      json.dump(rows, json_file, indent=2)
  else:
    dataframe = pl.DataFrame(rows)
    if output_path.endswith('.parquet'):
      dataframe.write_parquet(output_path)
    elif output_path.endswith('.tsv') or output_path.endswith('.txt'):
      dataframe.write_csv(output_path, separator='\t')
    else:
      dataframe.write_csv(output_path)


def _export_scored_variants(
    output_path: str,
    variants: Sequence[VariantRecord],
    *,
    include_features: bool = False,
    include_track_info: bool = False,
    include_rank: bool = False,
) -> None:
  """Exports scored variants to a tabular or JSON file."""
  rows = [
      v.to_row(
          include_features=include_features,
          include_track_info=include_track_info,
          rank=rank if include_rank else None,
      )
      for rank, v in enumerate(variants, start=1)
  ]
  _write_tabular(output_path, rows)


def _update_vcf_header(
    header: pysam.VariantHeader,
    scored_variants: Sequence[VariantRecord],
    *,
    vep_csq_only: bool,
) -> bool:
  """Updates VCF header with AVI info tags and CSQ schema."""
  for variant in scored_variants:
    contig_name = _normalize_chrom(variant.chromosome, with_chr_prefix=True)
    if contig_name not in header.contigs:
      header.contigs.add(contig_name)

  if not vep_csq_only:
    if 'AVI_PHRED' not in header.info:
      header.info.add(
          'AVI_PHRED',
          1,
          'Float',
          'AlphaGenome Variant Impact Phred-scaled Score',
      )
    if 'AVI_RAW' not in header.info:
      header.info.add(
          'AVI_RAW', 1, 'Float', 'AlphaGenome Raw Variant Impact Score'
      )
    if 'AVI_QUANTILE' not in header.info:
      header.info.add(
          'AVI_QUANTILE', 1, 'Float', 'AlphaGenome Tail Quantile Score'
      )
    if 'AVI_TOP_PERCENTILE' not in header.info:
      header.info.add(
          'AVI_TOP_PERCENTILE',
          1,
          'Float',
          'AlphaGenome Top Percentile of Genome-Wide SNVs',
      )
    if 'AVI_TOP_FEATURE' not in header.info:
      header.info.add(
          'AVI_TOP_FEATURE',
          1,
          'String',
          'AlphaGenome Top Contributing Modality',
      )

  has_existing_csq = 'CSQ' in header.info
  avi_csq_fields = [
      'AVI_PHRED',
      'AVI_RAW',
      'AVI_QUANTILE',
      'AVI_TOP_PERCENTILE',
      'AVI_TOP_FEATURE',
  ]

  if has_existing_csq:
    csq_desc = header.info['CSQ'].description or ''
    match = re.search(r'Format:\s*([^\\\">]+)', csq_desc)
    csq_format_fields = (
        match.group(1).split('|') if match else ['Allele', 'Consequence']
    )
    fields_to_add = [f for f in avi_csq_fields if f not in csq_format_fields]
    updated_format = '|'.join(csq_format_fields + fields_to_add)
    updated_desc = (
        f'Consequence annotations from Ensembl VEP. Format: {updated_format}'
    )
    header.info.remove_header('CSQ')
    header.add_line(
        f'##INFO=<ID=CSQ,Number=.,Type=String,Description="{updated_desc}">'
    )
  else:
    updated_format = 'Allele|' + '|'.join(avi_csq_fields)
    header.add_line(
        '##INFO=<ID=CSQ,Number=.,Type=String,Description="Consequence'
        f' annotations from Ensembl VEP. Format: {updated_format}">'
    )

  return has_existing_csq


def _write_annotated_vcf(
    input_path: str,
    output_path: str,
    scored_variants: Sequence[VariantRecord],
    *,
    vep_csq_only: bool = False,
) -> None:
  """Writes annotated variants to an output VCF."""
  with pysam.VariantFile(input_path) as vcf_in:
    has_existing_csq = _update_vcf_header(
        vcf_in.header, scored_variants, vep_csq_only=vep_csq_only
    )
    variant_map: Mapping[tuple[str, int, str, str], VariantRecord] = {
        (
            _normalize_chrom(v.chromosome),
            v.position,
            v.reference_bases,
            v.alternate_bases,
        ): v
        for v in scored_variants
    }

    with pysam.VariantFile(output_path, 'w', header=vcf_in.header) as vcf_out:
      for record in vcf_in.fetch():
        new_record = record.copy()
        chrom = _normalize_chrom(record.chrom)
        position = int(record.pos)
        ref = record.ref

        alt_avi_map: dict[str, str] = {}
        for alt in record.alts or ():
          variant = variant_map.get((chrom, position, ref, alt))
          phred_str = (
              f'{variant.phred_score:.3f}'
              if (variant and variant.phred_score is not None)
              else ''
          )
          raw_str = (
              f'{variant.raw_score:.4f}'
              if (variant and variant.raw_score is not None)
              else ''
          )
          quant_str = (
              f'{variant.quantile:.6f}'
              if (variant and variant.quantile is not None)
              else ''
          )
          pct_str = (
              f'{variant.top_percentile:.4f}'
              if (variant and variant.top_percentile is not None)
              else ''
          )
          mod_str = (
              variant.top_feature_name
              if (variant and variant.top_feature_name)
              else ''
          )

          if not vep_csq_only and variant:
            if variant.phred_score is not None:
              new_record.info['AVI_PHRED'] = round(variant.phred_score, 3)
            if variant.raw_score is not None:
              new_record.info['AVI_RAW'] = round(variant.raw_score, 4)
            if variant.quantile is not None:
              new_record.info['AVI_QUANTILE'] = round(variant.quantile, 6)
            if variant.top_percentile is not None:
              new_record.info['AVI_TOP_PERCENTILE'] = round(
                  variant.top_percentile, 4
              )
            if variant.top_feature_name:
              new_record.info['AVI_TOP_FEATURE'] = variant.top_feature_name

          alt_avi_map[alt] = (
              f'{phred_str}|{raw_str}|{quant_str}|{pct_str}|{mod_str}'
          )

        if has_existing_csq and 'CSQ' in new_record.info:
          current_csq_list = list(new_record.info['CSQ'])
          updated_csq_list = []
          for csq_item in current_csq_list:
            csq_tokens = csq_item.split('|')
            csq_allele = csq_tokens[0] if csq_tokens else ''
            avi_vals_str = alt_avi_map.get(csq_allele, '||||')
            updated_csq_list.append(f'{csq_item}|{avi_vals_str}')
          new_record.info['CSQ'] = tuple(updated_csq_list)
        else:
          csq_entries = [
              f'{alt}|{alt_avi_map[alt]}' for alt in record.alts or ()
          ]
          new_record.info['CSQ'] = tuple(csq_entries)

        vcf_out.write(new_record)


def print_top_summary_table(
    top_variants: Sequence[VariantRecord], total_count: int
) -> None:
  """Prints a summary markdown table of top candidate variants."""
  print(f'\n### Top Ranked AVI Variants (Total Analyzed: {total_count})\n')
  table_rows = []
  for rank, variant in enumerate(top_variants, start=1):
    phred_str = (
        f'{variant.phred_score:.2f}'
        if variant.phred_score is not None
        else 'N/A'
    )
    raw_str = (
        f'{variant.raw_score:.4f}' if variant.raw_score is not None else 'N/A'
    )
    if variant.top_percentile is not None:
      pct = variant.top_percentile
      pct_str = f'Top {pct:.2f}%' if pct >= 0.01 else f'Top {pct:.4g}%'
    else:
      pct_str = 'N/A'
    mod_str = variant.top_feature_name if variant.top_feature_name else 'Global'
    table_rows.append({
        'Rank': rank,
        'Variant': f'`{variant.variant_id}`',
        'AVI Phred': f'**{phred_str}**',
        'Raw Score': raw_str,
        'Top Percentile': pct_str,
        'Top Modality': mod_str,
    })
  print(tabulate.tabulate(table_rows, headers='keys', tablefmt='github'))
  print(
      '\n> [!NOTE] Phred = -10 * log10(quantile), representing the top'
      ' (10^(-Phred/10) * 100)% predicted impact of all genome-wide SNVs.\n'
  )


def print_feature_importance_breakdown(
    variant: VariantRecord,
    top_n: int = 10,
    *,
    include_track_info: bool = False,
) -> None:
  """Prints a detailed feature importance breakdown table for a variant."""
  if not variant.feature_importances:
    print(f'No feature importances available for {variant.variant_id}.')
    return

  sorted_features = sorted(
      variant.feature_importances.items(),
      key=lambda item: abs(item[1]),
      reverse=True,
  )

  phred_display = (
      f'{variant.phred_score:.2f}' if variant.phred_score is not None else 'N/A'
  )
  if variant.top_percentile is not None:
    if variant.top_percentile >= 0.01:
      pct_info = f' | Top {variant.top_percentile:.2f}%'
    else:
      pct_info = f' | Top {variant.top_percentile:.4g}%'
  else:
    pct_info = ''
  print(
      f'\n#### Feature Importances for `{variant.variant_id}` (AVI Phred:'
      f' {phred_display}{pct_info})'
  )

  max_abs = abs(sorted_features[0][1]) if sorted_features else 1.0
  max_abs = max(max_abs, 1e-9)

  if include_track_info and variant.track_attributions:
    table_rows = []
    for rank, (feat_name, feat_val) in enumerate(
        sorted_features[:top_n], start=1
    ):
      display_name = format_feature_name(feat_name)
      bar_length = int(round((abs(feat_val) / max_abs) * 15))
      bar_str = '#' * bar_length
      attr = variant.track_attributions.get(feat_name)
      if attr:
        track_idx_str = (
            f'`{attr.track_index}`' if attr.track_index >= 0 else 'N/A'
        )
        top_info = (
            f'{attr.biosample_name} ({attr.track_name})'
            if (attr.biosample_name and attr.biosample_name != 'N/A')
            else attr.track_name
        )
        gene_str = attr.gene_name if attr.gene_name else 'N/A'
        table_rows.append({
            'Rank': rank,
            'Modality / Feature': f'`{display_name}`',
            'Attribution': f'`{feat_val:+.4f}`',
            'Track Index': track_idx_str,
            'Top Track / Biosample': top_info,
            'Target Gene': gene_str,
            'Relative Impact': f'`{bar_str}`',
        })
      else:
        table_rows.append({
            'Rank': rank,
            'Modality / Feature': f'`{display_name}`',
            'Attribution': f'`{feat_val:+.4f}`',
            'Track Index': 'N/A',
            'Top Track / Biosample': 'N/A',
            'Target Gene': 'N/A',
            'Relative Impact': f'`{bar_str}`',
        })
    print(tabulate.tabulate(table_rows, headers='keys', tablefmt='github'))
  else:
    table_rows = []
    for rank, (feat_name, feat_val) in enumerate(
        sorted_features[:top_n], start=1
    ):
      display_name = format_feature_name(feat_name)
      bar_length = int(round((abs(feat_val) / max_abs) * 15))
      bar_str = '#' * bar_length
      table_rows.append({
          'Rank': rank,
          'Modality / Feature': f'`{display_name}`',
          'Attribution Weight': f'`{feat_val:+.4f}`',
          'Relative Impact': f'`{bar_str}`',
      })
    print(tabulate.tabulate(table_rows, headers='keys', tablefmt='github'))
  print()


def _print_feature_breakdowns(
    variants: Sequence[VariantRecord],
    *,
    header_text: str = '',
    top_n: int = 10,
    include_track_info: bool = False,
    max_variants: int | None = None,
) -> None:
  """Prints feature importance breakdowns for a list of variants."""
  if not variants:
    return
  if header_text:
    print(header_text)
  display = variants[:max_variants] if max_variants is not None else variants
  for variant in display:
    print_feature_importance_breakdown(
        variant, top_n=top_n, include_track_info=include_track_info
    )


def _extract_region_variants(
    avi_adata: anndata.AnnData,
    fi_adata: anndata.AnnData | None,
) -> Sequence[VariantRecord]:
  """Extracts scored VariantRecord instances from interval query AnnData."""
  variant_objs = list(avi_adata.obs['variant'])
  raw_scores = np.ravel(avi_adata.X) if avi_adata.X is not None else None
  quantiles = (
      np.ravel(avi_adata.layers['quantiles'])
      if 'quantiles' in avi_adata.layers
      else None
  )

  fi_matrix = (
      fi_adata.X if (fi_adata is not None and fi_adata.X is not None) else None
  )
  fi_names: Sequence[str] = ()
  if fi_adata is not None:
    fi_names = _resolve_feature_names(fi_adata)

  has_valid_fi = (
      fi_matrix is not None
      and bool(fi_names)
      and fi_matrix.shape[1] == len(fi_names)
  )

  candidate_variants: MutableSequence[VariantRecord] = []
  for index, variant_obj in enumerate(variant_objs):
    base = _to_variant_record(variant_obj)
    raw_val = float(raw_scores[index]) if raw_scores is not None else None
    phred_val: float | None = None
    quantile_val: float | None = None

    if quantiles is not None:
      quantile_val, phred_val = _cdf_to_phred(float(quantiles[index]))

    fi_dict: dict[str, float] = {}
    top_feature_name: str | None = None
    top_feature_value: float | None = None

    if has_valid_fi and fi_matrix is not None:
      row_fi = np.ravel(fi_matrix[index])
      fi_dict, top_feature_name, top_feature_value = (
          _compute_feature_attributions(row_fi, fi_names)
      )

    candidate_variants.append(
        dataclasses.replace(
            base,
            raw_score=raw_val,
            phred_score=phred_val,
            quantile=quantile_val,
            top_feature_name=top_feature_name,
            top_feature_value=top_feature_value,
            feature_importances=fi_dict,
        )
    )

  return candidate_variants


def handle_annotate(args: argparse.Namespace) -> None:
  """Handler for `annotate` subcommand."""
  client = get_atlas_client()

  is_vcf = (
      args.input.endswith('.vcf')
      or args.input.endswith('.vcf.gz')
      or args.input.endswith('.bcf')
  )
  if is_vcf:
    variants, _ = _read_variants_vcf(args.input)
  else:
    variants = _read_variants_tabular(args.input)

  if not variants:
    print(f'No variants found in input file: {args.input}')
    return

  print(
      f'Loaded {len(variants)} variants. Scoring via AlphaGenome Atlas API...'
  )
  scored = score_variant_batch(
      variants, client, include_track_info=args.include_track_info
  )
  ranked, top_hits = _rank_and_filter(
      scored, min_phred=args.min_phred, top_k=args.top_k
  )

  if is_vcf and (
      args.output.endswith('.vcf') or args.output.endswith('.vcf.gz')
  ):
    _write_annotated_vcf(
        args.input,
        args.output,
        ranked,
        vep_csq_only=args.vep_csq_only,
    )
  else:
    _export_scored_variants(
        args.output,
        ranked,
        include_features=args.include_features,
        include_track_info=args.include_track_info,
    )

  print(f'Annotated variants successfully written to: {args.output}')

  print_top_summary_table(top_hits, len(ranked))

  if args.show_features and top_hits:
    _print_feature_breakdowns(
        top_hits,
        header_text='\n### Top Candidate Modality Attribution Breakdowns:',
        include_track_info=args.include_track_info,
        max_variants=3,
    )

  if args.top_output:
    _export_scored_variants(
        args.top_output,
        top_hits,
        include_features=args.include_features,
        include_track_info=args.include_track_info,
        include_rank=True,
    )
    print(f'Top variants summary written to: {args.top_output}')


def handle_query(args: argparse.Namespace) -> None:
  """Handler for `query` subcommand."""
  client = get_atlas_client()

  raw_variant_strings: MutableSequence[str] = []
  if args.variants:
    raw_variant_strings.extend(args.variants)

  if args.input_file:
    with open(args.input_file, 'r', encoding='utf-8') as input_file:
      for line in input_file:
        line_str = line.strip()
        if line_str and not line_str.startswith('#'):
          raw_variant_strings.append(line_str)

  if not raw_variant_strings:
    print('No variants specified. Use positional arguments or --input_file.')
    return

  variants: MutableSequence[VariantRecord] = []
  for variant_str in raw_variant_strings:
    try:
      parsed = genome.Variant.from_str(variant_str.strip())
      variants.append(_to_variant_record(parsed))
    except ValueError:
      sys.stderr.write(
          f'Invalid variant format: {variant_str}. Expected chr:pos:ref>alt.\n'
      )

  if not variants:
    return

  scored = score_variant_batch(
      variants, client, include_track_info=args.include_track_info
  )

  if args.format == 'json':
    out_rows = [
        v.to_row(
            include_features=True,
            include_track_info=args.include_track_info,
        )
        for v in scored
    ]
    json_str = json.dumps(out_rows, indent=2)
    if args.output:
      with open(args.output, 'w', encoding='utf-8') as out_f:
        out_f.write(json_str)
      print(f'Results written to {args.output}')
    else:
      print(json_str)
    return

  print_top_summary_table(scored, len(scored))

  if args.show_features:
    for variant in scored:
      print_feature_importance_breakdown(
          variant,
          top_n=args.top_features,
          include_track_info=args.include_track_info,
      )

  if args.output:
    _export_scored_variants(
        args.output,
        scored,
        include_features=args.include_features,
        include_track_info=args.include_track_info,
    )
    print(f'Results exported to: {args.output}')


def handle_region(args: argparse.Namespace) -> None:
  """Handler for `region` subcommand (in silico saturation mutagenesis)."""
  client = get_atlas_client()

  region_match = re.match(
      r'^(chr[0-9XYM]+|\d+|X|Y|M):(\d+)-(\d+)$', args.region
  )
  if not region_match:
    raise ValueError(
        f'Invalid region string: {args.region}. Expected format chr:start-end'
        ' (e.g. chr9:128225900-128226000).'
    )

  chrom = region_match.group(1)
  start_pos = int(region_match.group(2))
  end_pos = int(region_match.group(3))

  if end_pos < start_pos:
    raise ValueError(
        f'End position ({end_pos}) must be >= start position ({start_pos}).'
    )

  window_size = end_pos - start_pos + 1
  if window_size > args.max_window_size:
    raise ValueError(
        f'Requested region size ({window_size} bp) exceeds maximum window size'
        f' ({args.max_window_size} bp). Please specify a smaller region or'
        ' increase --max_window_size.'
    )

  print(f'Scanning region {chrom}:{start_pos}-{end_pos} ({window_size} bp)...')

  chrom_normalized = _normalize_chrom(chrom, with_chr_prefix=True)
  # Convert 1-based closed CLI interval [start, end] to 0-based half-open [start-1, end)
  interval = genome.Interval(
      chromosome=chrom_normalized,
      start=start_pos - 1,
      end=end_pos,
  )

  print(
      'Querying AlphaGenome Atlas interval API for'
      f' {chrom_normalized}:{start_pos}-{end_pos}...'
  )
  scores = client.query_interval(interval, requested_scorers=_REQUESTED_SCORERS)

  if 'AVI_SCORE' not in scores:
    raise ValueError(
        f'AlphaGenome Atlas did not return AVI_SCORE for region {args.region}.'
    )

  avi_adata = scores['AVI_SCORE']
  fi_adata = scores.get('AVI_SCORE_FEATURE_IMPORTANCE')

  candidate_variants = _extract_region_variants(avi_adata, fi_adata)
  valid_scored = [
      v
      for v in candidate_variants
      if v.phred_score is not None or v.raw_score is not None
  ]
  print(
      f'Successfully retrieved and scored {len(valid_scored)} SNVs in region.'
  )

  ranked, top_hits = _rank_and_filter(
      valid_scored, min_phred=args.min_phred, top_k=args.top_k
  )

  print_top_summary_table(top_hits, len(valid_scored))

  if args.show_features and top_hits:
    _print_feature_breakdowns(
        top_hits,
        header_text='\n### Top Region Hotspot Feature Importances:',
        include_track_info=args.include_track_info,
        max_variants=3,
    )

  if args.output:
    _export_scored_variants(
        args.output,
        ranked,
        include_features=args.include_features,
        include_track_info=args.include_track_info,
        include_rank=True,
    )
    print(f'Region scan exported to: {args.output}')


def _print_feature_metadata(args: argparse.Namespace) -> None:
  """Prints AVI feature definitions (no API client needed)."""
  rows: list[dict[str, object]] = []
  for rank, feature in enumerate(AviFeature, start=1):
    rows.append({
        'rank': rank,
        'feature_key': feature.name,
        'display_name': feature.display_name,
        'category': feature.category,
    })

  if getattr(args, 'format', 'table') == 'json':
    print(json.dumps(rows, indent=2))
    return
  if getattr(args, 'output', None):
    _write_tabular(args.output, rows)
    print(f'AVI feature definitions written to: {args.output}')
    return

  print('\n### AlphaGenome Variant Impact (AVI) 18 Biological Modalities:\n')
  display_rows = [
      {
          'Rank': r['rank'],
          'Modality Key': f'`{r["feature_key"]}`',
          'Website Display Name': f'**{r["display_name"]}**',
          'Category': r['category'],
      }
      for r in rows
  ]
  print(tabulate.tabulate(display_rows, headers='keys', tablefmt='github'))
  print()


def handle_metadata(args: argparse.Namespace) -> None:
  """Handler for `metadata` subcommand."""
  meta_type = args.type

  if meta_type == 'features':
    _print_feature_metadata(args)
    return

  # scorers and tracks modes require the Atlas client.
  client = get_atlas_client()

  if meta_type == 'scorers':
    scorer_map = client.scorer_metadata()
    rows: MutableSequence[dict[str, object]] = [
        {
            'scorer_name': 'AVI_SCORE',
            'category': 'Variant Impact (Composite)',
            'is_signed': False,
            'tracks': 1,
            'biosamples': 1,
            'description': (
                'AlphaGenome Variant Impact calibrated Phred/Quantile score'
            ),
        },
        {
            'scorer_name': 'AVI_SCORE_FEATURE_IMPORTANCE',
            'category': 'Feature Attribution',
            'is_signed': True,
            'tracks': 18,
            'biosamples': 1,
            'description': (
                '18-modality biological feature attribution weights (SHAP)'
            ),
        },
    ]
    for name, meta in scorer_map.items():
      df = meta.track_metadata
      num_tracks = len(df)
      num_biosamples = (
          int(df['biosample_name'].nunique())
          if (df is not None and 'biosample_name' in df.columns)
          else 0
      )
      rows.append({
          'scorer_name': name,
          'category': 'Experimental Track Scorer',
          'is_signed': meta.is_signed,
          'tracks': num_tracks,
          'biosamples': num_biosamples,
          'description': f'Atlas experimental track predictions ({name})',
      })

    if args.format == 'json':
      print(json.dumps(rows, indent=2))
      return
    if args.output:
      _write_tabular(args.output, rows)
      print(f'Scorer metadata written to: {args.output}')
      return

    print('\n### Registered AlphaGenome Atlas Scorers:\n')
    display_rows = [
        {
            'Scorer Name': f'`{row["scorer_name"]}`',
            'Category': row['category'],
            'Signed': row['is_signed'],
            'Tracks': row['tracks'],
            'Biosamples': row['biosamples'],
        }
        for row in rows
    ]
    print(tabulate.tabulate(display_rows, headers='keys', tablefmt='github'))
    print()

  elif meta_type == 'tracks':
    scorer_map = client.scorer_metadata()
    track_rows: MutableSequence[dict[str, object]] = []

    target_scorers = [args.scorer] if args.scorer else list(scorer_map.keys())

    for name in target_scorers:
      if name not in scorer_map:
        continue
      df = scorer_map[name].track_metadata
      if df is None or df.empty:
        continue
      for index, row in df.iterrows():
        track_dict: dict[str, object] = {
            'scorer': name,
            'track_index': int(index),
            'track_name': str(row.get('name', '')),
            'biosample_name': str(row.get('biosample_name', '')),
            'biosample_type': str(row.get('biosample_type', '')),
            'ontology_curie': str(row.get('ontology_curie', '')),
            'strand': str(row.get('strand', '')),
        }
        for extra_col in ('transcription_factor', 'histone_mark', 'assay'):
          if extra_col in row and str(row[extra_col]):
            track_dict[extra_col] = str(row[extra_col])

        # Filter by ontology
        if args.ontology and str(track_dict['ontology_curie']) != args.ontology:
          continue

        # Filter by text query
        if args.query:
          q_lower = args.query.lower()
          searchable = ' '.join(str(v).lower() for v in track_dict.values())
          if q_lower not in searchable:
            continue

        track_rows.append(track_dict)

    if not track_rows:
      print('No tracks matched the specified metadata filter criteria.')
      return

    if args.output:
      _write_tabular(args.output, track_rows)
      print(
          f'Dumped {len(track_rows)} track metadata records to: {args.output}'
      )
      return

    if args.format == 'json':
      print(json.dumps(track_rows[: args.top_n], indent=2))
      return

    print(
        f'\n### Atlas Track Metadata Catalog (Matched: {len(track_rows)},'
        f' Showing Top {min(len(track_rows), args.top_n)}):\n'
    )
    display_tracks = [
        {
            'Scorer': f'`{row["scorer"]}`',
            'Track Index': f'`{row["track_index"]}`',
            'Track Name': f'`{row["track_name"]}`',
            'Biosample': row['biosample_name'],
            'Ontology': f'`{row["ontology_curie"]}`',
        }
        for row in track_rows[: args.top_n]
    ]
    print(tabulate.tabulate(display_tracks, headers='keys', tablefmt='github'))
    print()


def _clean_chrom(chrom_str: str) -> str:
  """Ensures chromosome string has chr prefix."""
  c = str(chrom_str).strip()
  return c if c.startswith('chr') else f'chr{c}'


GTF_URL = (
    'https://storage.googleapis.com/alphagenome/reference/gencode/'
    'hg38/gencode.v46.annotation.gtf.gz.feather'
)


def load_gtf_dataframe(custom_path: str | None = None) -> pd.DataFrame:
  """Loads GENCODE v46 GTF annotations from a local path, environment variable, cache, or GCS."""
  path = custom_path or os.environ.get('ALPHAGENOME_GTF_PATH')
  if path and os.path.exists(path):
    print(f'Loading GTF from {path}...')
    return pd.read_feather(path)

  cache_dir = os.path.expanduser('~/.cache/alphagenome')
  cache_path = os.path.join(cache_dir, 'gencode.v46.annotation.gtf.gz.feather')
  if os.path.exists(cache_path):
    print(f'Loading GTF from cache {cache_path}...')
    return pd.read_feather(cache_path)

  target_url = path or GTF_URL
  print(f'Loading GTF from {target_url}...')
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
        f'Failed to load GTF annotations from {target_url}: {e}'
    ) from e


def handle_gtf(args: argparse.Namespace) -> None:
  """Handles gene and transcript GTF queries, exon extraction, and splice junctions."""
  gtf = load_gtf_dataframe()

  if getattr(args, 'protein_coding', False):
    if 'gene_type' in gtf.columns:
      gtf = gtf[gtf['gene_type'] == 'protein_coding']
    if 'transcript_type' in gtf.columns:
      gtf = gtf[gtf['transcript_type'] == 'protein_coding']

  if not getattr(args, 'all_transcripts', False):
    if 'tag' in gtf.columns:
      gtf = gtf[gtf['tag'].fillna('').str.contains('MANE_Select')]

  if getattr(args, 'longest', False):
    lengths = gtf[gtf['Feature'] == 'transcript'].reset_index(drop=True)
    lengths['t_len'] = lengths['End'] - lengths['Start'] + 1
    longest = lengths.loc[lengths.groupby('gene_id')['t_len'].idxmax()]
    gtf = gtf[gtf['transcript_id'].isin(longest['transcript_id'])]

  query_variant = None
  query_pos = None
  query_chrom = None

  if getattr(args, 'variant', None):
    v_str = args.variant.strip()
    match = re.match(
        r'^(chr[0-9XYM]+):(\d+)(?:[:/_>]([A-Za-z]+)(?:[:/_>]([A-Za-z]+))?)?$',
        v_str,
    )
    if match:
      query_chrom = match.group(1)
      query_pos = int(match.group(2))
      query_variant = v_str
    else:
      parts = v_str.split(':')
      if len(parts) >= 2 and parts[0].startswith('chr') and parts[1].isdigit():
        query_chrom = parts[0]
        query_pos = int(parts[1])
        query_variant = v_str
      else:
        raise ValueError(f'Invalid variant/position format: {args.variant}')

    t_df = gtf[
        (gtf['Chromosome'] == query_chrom) & (gtf['Feature'] == 'transcript')
    ]
    overlapping_t = t_df[
        (t_df['Start'] <= query_pos) & (t_df['End'] >= query_pos)
    ]
    if overlapping_t.empty:
      nearby_t = t_df[
          (t_df['Start'] - 50000 <= query_pos)
          & (t_df['End'] + 50000 >= query_pos)
      ]
      if nearby_t.empty:
        print(f'No transcripts found overlapping or near {args.variant}.')
        return
      t_ids = nearby_t['transcript_id'].unique()
    else:
      t_ids = overlapping_t['transcript_id'].unique()
    target_rows = gtf[gtf['transcript_id'].isin(t_ids)]

  elif getattr(args, 'gene', None):
    gene_name = args.gene.strip().upper()
    target_rows = gtf[gtf['gene_name'].fillna('').str.upper() == gene_name]
    if target_rows.empty:
      print(f'No gene found with symbol: {args.gene}')
      return

  elif getattr(args, 'gene_id', None):
    gene_id = args.gene_id.strip()
    target_rows = gtf[
        (gtf['gene_id'] == gene_id)
        | (gtf['gene_id'].str.split('.').str[0] == gene_id.split('.')[0])
    ]
    if target_rows.empty:
      print(f'No gene found with ID: {args.gene_id}')
      return

  elif getattr(args, 'transcript_id', None):
    t_id = args.transcript_id.strip()
    target_rows = gtf[
        (gtf['transcript_id'] == t_id)
        | (gtf['transcript_id'].str.split('.').str[0] == t_id.split('.')[0])
    ]
    if target_rows.empty:
      print(f'No transcript found with ID: {args.transcript_id}')
      return

  elif getattr(args, 'region', None):
    r_str = args.region.strip()
    match = re.match(r'^(chr[0-9XYM]+):(\d+)-(\d+)$', r_str)
    if not match:
      raise ValueError(f'Invalid region format: {args.region}')
    r_chrom = match.group(1)
    r_start = int(match.group(2))
    r_end = int(match.group(3))
    t_df = gtf[
        (gtf['Chromosome'] == r_chrom) & (gtf['Feature'] == 'transcript')
    ]
    overlapping_t = t_df[(t_df['Start'] <= r_end) & (t_df['End'] >= r_start)]
    if overlapping_t.empty:
      print(f'No transcripts found overlapping region {args.region}.')
      return
    target_rows = gtf[
        gtf['transcript_id'].isin(overlapping_t['transcript_id'].unique())
    ]

  else:
    raise ValueError(
        'Must specify one of --gene, --gene_id, --transcript_id, --variant, or'
        ' --region.'
    )

  transcripts_data = []
  for t_id, group in target_rows.groupby('transcript_id'):
    t_feat_rows = group[group['Feature'] == 'transcript']
    t_feat = t_feat_rows.iloc[0] if not t_feat_rows.empty else group.iloc[0]
    gene_name = str(t_feat.get('gene_name', ''))
    gene_id = str(t_feat.get('gene_id', ''))
    chrom = str(t_feat.get('Chromosome', ''))
    strand = str(t_feat.get('Strand', ''))
    t_start = int(t_feat.get('Start', 0))
    t_end = int(t_feat.get('End', 0))
    t_type = str(t_feat.get('transcript_type', ''))
    tags = str(t_feat.get('tag', ''))
    is_mane = 'MANE_Select' in tags

    ascending = strand == '+'
    exon_rows = (
        group[group['Feature'] == 'exon']
        .sort_values('Start', ascending=ascending)
        .reset_index(drop=True)
    )
    exons_list = []
    for idx, e in exon_rows.iterrows():
      e_num = int(e.get('exon_number', idx + 1))
      e_start = int(e['Start'])
      e_end = int(e['End'])
      e_width = e_end - e_start + 1
      acceptor = e_start if strand == '+' else e_end
      donor = e_end if strand == '+' else e_start
      overlaps_v = False
      if query_pos is not None and chrom == query_chrom:
        overlaps_v = e_start <= query_pos <= e_end

      exons_list.append({
          'exon_number': e_num,
          'chromosome': chrom,
          'start': e_start,
          'end': e_end,
          'width': e_width,
          'strand': strand,
          'acceptor': acceptor,
          'donor': donor,
          'overlaps_variant': overlaps_v,
      })
    if exon_rows.empty:
      print('No exons found for target.', file=sys.stderr)
      sys.exit(1)

    junctions_list = []
    num_exons = len(exon_rows)
    for i in range(num_exons - 1):
      e1 = exon_rows.iloc[i]
      e2 = exon_rows.iloc[i + 1]
      if strand == '+':
        j_start = int(e1['End'])
        j_end = int(e2['Start'])
      else:
        j_start = int(e2['End'])
        j_end = int(e1['Start'])
      score_id = None
      atlas_url = None
      if query_variant:
        v_enc = query_variant.replace('>', '%3E')
        track_name = (
            getattr(args, 'track', None)
            or 'junction_UBERON:0011907 gtex Muscle_Skeletal polyA plus RNA-seq'
        )
        track_enc = track_name.replace(' ', '%20')
        splice_score_id = (
            f'{v_enc}:SPLICE_JUNCTIONS:{track_enc}:Unstranded:'
            f'{gene_id}:{j_start}:{j_end}'
        )
        # Splicing visualization: pair SPLICE_JUNCTIONS with continuous RNA_SEQ
        rna_track_name = re.sub(r'^junction_', '', track_name)
        rna_track_enc = rna_track_name.replace(' ', '%20')
        rna_score_id = (
            f'{v_enc}:RNA_SEQ:{rna_track_enc}:Unstranded:{gene_id}:0:0'
        )
        scores_param = f'{splice_score_id},{rna_score_id}'
        score_id = splice_score_id
        i_start = max(1, t_start)
        i_end = t_end
        atlas_url = (
            f'{_ATLAS_TRACK_PREDICTIONS_URL}?'
            f'q={v_enc}&m=variant'
            '&lItems=avi,section:RNA_SEQ,section:SPLICE_JUNCTIONS,section:SPLICE_SITE_USAGE,section:SPLICE_SITES'
            f'&scores={scores_param}'
            f'&i={chrom}:{i_start}-{i_end}'
        )

      junctions_list.append({
          'type': f'Canonical Intron {i+1}',
          'upstream_exon': int(e1.get('exon_number', i + 1)),
          'downstream_exon': int(e2.get('exon_number', i + 2)),
          'junction_start': j_start,
          'junction_end': j_end,
          'junction_str': f'{j_start}:{j_end}',
          'intron_length': j_end - j_start - 1,
          'score_id': score_id,
          'atlas_url': atlas_url,
      })

    for i in range(num_exons - 2):
      e1 = exon_rows.iloc[i]
      e_skipped = exon_rows.iloc[i + 1]
      e3 = exon_rows.iloc[i + 2]
      if strand == '+':
        j_start = int(e1['End'])
        j_end = int(e3['Start'])
      else:
        j_start = int(e3['End'])
        j_end = int(e1['Start'])

      junctions_list.append({
          'type': f'Exon {e_skipped.get("exon_number", i+2)} Skipping',
          'upstream_exon': int(e1.get('exon_number', i + 1)),
          'downstream_exon': int(e3.get('exon_number', i + 3)),
          'junction_start': j_start,
          'junction_end': j_end,
          'junction_str': f'{j_start}:{j_end}',
          'intron_length': j_end - j_start - 1,
      })

    cds_rows = group[group['Feature'] == 'CDS'].sort_values(
        'Start', ascending=ascending
    )
    cds_list = [
        {
            'start': int(r['Start']),
            'end': int(r['End']),
            'width': int(r['End']) - int(r['Start']) + 1,
        }
        for _, r in cds_rows.iterrows()
    ]

    utr_rows = group[group['Feature'] == 'UTR'].sort_values(
        'Start', ascending=ascending
    )
    utr_list = [
        {
            'start': int(r['Start']),
            'end': int(r['End']),
            'width': int(r['End']) - int(r['Start']) + 1,
        }
        for _, r in utr_rows.iterrows()
    ]

    transcripts_data.append({
        'gene_name': gene_name,
        'gene_id': gene_id,
        'transcript_id': t_id,
        'transcript_type': t_type,
        'chromosome': chrom,
        'start': t_start,
        'end': t_end,
        'strand': strand,
        'is_mane_select': is_mane,
        'num_exons': len(exons_list),
        'exons': exons_list,
        'junctions': junctions_list,
        'cds': cds_list,
        'utr': utr_list,
    })

  if not transcripts_data:
    print('No matching transcript records found.')
    return

  if getattr(args, 'format', 'table') == 'json':
    print(json.dumps(transcripts_data, indent=2))
    return

  if getattr(args, 'output', None):
    # Flatten exons or junctions for output export
    export_rows = []
    for t in transcripts_data:
      if getattr(args, 'junctions', False):
        for j in t['junctions']:
          row = {
              'gene_name': t['gene_name'],
              'gene_id': t['gene_id'],
              'transcript_id': t['transcript_id'],
              **j,
          }
          export_rows.append(row)
      else:
        for e in t['exons']:
          row = {
              'gene_name': t['gene_name'],
              'gene_id': t['gene_id'],
              'transcript_id': t['transcript_id'],
              **e,
          }
          export_rows.append(row)
    _write_tabular(args.output, export_rows)
    print(f'Exported {len(export_rows)} records to: {args.output}')
    return

  for t in transcripts_data:
    mane_badge = ' [MANE Select]' if t['is_mane_select'] else ''
    print(
        f'\n### Gene: {t["gene_name"]} ({t["gene_id"]}) | Transcript:'
        f' {t["transcript_id"]}{mane_badge}'
    )
    print(
        f'Coordinates: {t["chromosome"]}:{t["start"]}-{t["end"]} ({t["strand"]}'
        f' strand) | Type: {t["transcript_type"]} | Exons: {t["num_exons"]}\n'
    )

    if getattr(args, 'exons', False) or (
        not getattr(args, 'junctions', False)
        and not getattr(args, 'cds', False)
        and not getattr(args, 'utr', False)
    ):
      print('#### Exon Boundaries:')
      exon_rows = [
          {
              'Exon #': f'Exon {e["exon_number"]}',
              'Coordinates': f'`{e["start"]}-{e["end"]}`',
              'Width (bp)': f'{e["width"]} bp',
              "5' Acceptor": f'`{e["acceptor"]}`',
              "3' Donor": f'`{e["donor"]}`',
              'Overlaps Query': (
                  '**Yes (Mutation Site)**' if e['overlaps_variant'] else '-'
              ),
          }
          for e in t['exons']
      ]
      print(tabulate.tabulate(exon_rows, headers='keys', tablefmt='github'))
      print()

    if getattr(args, 'junctions', False):
      print('#### Splice Junctions (Canonical & Exon Skipping):')
      junction_rows = [
          {
              'Junction Type': j['type'],
              'Flanking Exons': (
                  f'Exon {j["upstream_exon"]} → Exon {j["downstream_exon"]}'
              ),
              'Junction Coordinates': f'`{j["junction_str"]}`',
              'Intron Length': f'{j["intron_length"]} bp',
          }
          for j in t['junctions']
      ]
      print(tabulate.tabulate(junction_rows, headers='keys', tablefmt='github'))
      print()

    if getattr(args, 'cds', False):
      print('#### CDS Segments:')
      for idx, c in enumerate(t['cds'], start=1):
        print(
            f'- CDS {idx}: `{c["start"]}-{c["end"]}` (Width: {c["width"]} bp)'
        )
      print()

    if getattr(args, 'utr', False):
      print('#### UTR Segments:')
      for idx, u in enumerate(t['utr'], start=1):
        print(
            f'- UTR {idx}: `{u["start"]}-{u["end"]}` (Width: {u["width"]} bp)'
        )
      print()


def _add_common_args(parser: argparse.ArgumentParser) -> None:
  """Adds shared CLI arguments across all subcommands."""
  parser.add_argument(
      '--show_features',
      '--features',
      dest='show_features',
      action=argparse.BooleanOptionalAction,
      default=True,
      help='Display feature importance breakdowns',
  )
  parser.add_argument(
      '--include_features',
      action=argparse.BooleanOptionalAction,
      default=False,
      help='Include all 18 feature importance columns in exported files',
  )
  parser.add_argument(
      '--include_track_info',
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          'Resolve and include the exact underlying Atlas track index,'
          ' biosample, and target gene for each feature'
      ),
  )
  parser.add_argument('--output', '-o', help='Path to write output file')


def main() -> None:
  dotenv.load_dotenv(os.path.expanduser('~/.env'))

  parser = argparse.ArgumentParser(
      description='AlphaGenome Variant Impact (AVI) CLI',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  subparsers = parser.add_subparsers(dest='subcommand', required=True)

  # Subcommand: annotate
  parser_annotate = subparsers.add_parser(
      'annotate',
      help=(
          'Annotate variant callset (VCF, CSV, TSV, Parquet) in Ensembl VEP'
          ' CSQ format'
      ),
  )
  parser_annotate.add_argument(
      '--input',
      '-i',
      required=True,
      help=(
          'Path to input variant file (VCF or tabular CSV/TSV/Parquet with'
          ' 1-based coordinates)'
      ),
  )
  _add_common_args(parser_annotate)
  parser_annotate.add_argument(
      '--top_k',
      type=int,
      default=20,
      help='Number of top variants to summarize',
  )
  parser_annotate.add_argument(
      '--min_phred',
      type=float,
      default=0.0,
      help='Minimum Phred score for top filtering',
  )
  parser_annotate.add_argument(
      '--top_output',
      help='Optional path to export top-ranked summary (JSON/TSV/CSV)',
  )
  parser_annotate.add_argument(
      '--vep_csq_only',
      action='store_true',
      help='Only annotate within VEP CSQ tag',
  )

  # Subcommand: query
  parser_query = subparsers.add_parser(
      'query',
      help=(
          'Query one or more variant strings to get scores and full modality'
          ' feature importances'
      ),
  )
  parser_query.add_argument(
      'variants',
      nargs='*',
      help=(
          'Variant strings to score in 1-based chr:pos:ref>alt format (e.g.'
          ' chr9:128225994:G>A chr22:36201698:A>C)'
      ),
  )
  parser_query.add_argument(
      '--input_file',
      '-i',
      help=(
          'Path to file containing variant strings in 1-based chr:pos:ref>alt'
          ' format (one per line)'
      ),
  )
  _add_common_args(parser_query)
  parser_query.add_argument(
      '--top_features',
      type=int,
      default=10,
      help='Number of top features to show in breakdown table',
  )
  parser_query.add_argument(
      '--format',
      choices=['table', 'json', 'tsv'],
      default='table',
      help='Output format',
  )

  # Subcommand: region
  parser_region = subparsers.add_parser(
      'region',
      help='In silico saturation mutagenesis scan across a genomic window',
  )
  parser_region.add_argument(
      '--region',
      '-r',
      required=True,
      help=(
          'Genomic interval in 1-based closed chr:start-end format (e.g.'
          ' chr9:128225900-128226050)'
      ),
  )
  _add_common_args(parser_region)
  parser_region.add_argument(
      '--top_k',
      type=int,
      default=20,
      help='Number of top variants to summarize',
  )
  parser_region.add_argument(
      '--min_phred',
      type=float,
      default=0.0,
      help='Minimum Phred score threshold for reporting',
  )
  parser_region.add_argument(
      '--max_window_size',
      type=int,
      default=1000,
      help='Maximum permitted window size in basepairs',
  )

  # Subcommand: metadata
  parser_metadata = subparsers.add_parser(
      'metadata',
      help=(
          'Dump and search Atlas database scorers, track catalogs, and AVI'
          ' feature attribution modalities'
      ),
  )
  parser_metadata.add_argument(
      '--type',
      choices=['scorers', 'tracks', 'features'],
      default='scorers',
      help='Type of metadata to dump or search',
  )
  parser_metadata.add_argument(
      '--scorers',
      action='store_const',
      const='scorers',
      dest='type',
      help='Shortcut for --type scorers',
  )
  parser_metadata.add_argument(
      '--tracks',
      action='store_const',
      const='tracks',
      dest='type',
      help='Shortcut for --type tracks',
  )
  parser_metadata.add_argument(
      '--features',
      action='store_const',
      const='features',
      dest='type',
      help='Shortcut for --type features',
  )
  parser_metadata.add_argument(
      '--scorer',
      help='Filter tracks by Atlas scorer name (e.g. CHIP_TF, RNA_SEQ)',
  )
  parser_metadata.add_argument(
      '--ontology',
      help='Filter tracks by ontology CURIE (e.g. UBERON:0002107, CL:0002545)',
  )
  parser_metadata.add_argument(
      '--query',
      '-q',
      help='Search text across track names, biosamples, and assay targets',
  )
  parser_metadata.add_argument(
      '--top_n',
      type=int,
      default=50,
      help='Maximum rows to print in table preview',
  )
  parser_metadata.add_argument(
      '--format',
      choices=['table', 'json', 'tsv'],
      default='table',
      help='Output format for stdout',
  )
  parser_metadata.add_argument(
      '--output',
      '-o',
      help='Optional path to export metadata (.parquet, .tsv, .csv, .json)',
  )

  # Subcommand: gtf
  parser_gtf = subparsers.add_parser(
      'gtf',
      help=(
          'Query GENCODE v46 gene annotations, extract exons, CDS, UTRs, and'
          ' compute splice junction coordinates'
      ),
  )
  parser_gtf.add_argument(
      '--gene',
      help='Gene symbol (e.g. CAPN3, NAGS, TERT, DNM1)',
  )
  parser_gtf.add_argument(
      '--gene_id',
      help='Ensembl Gene ID (e.g. ENSG00000092529 or ENSG00000092529.26)',
  )
  parser_gtf.add_argument(
      '--transcript_id',
      help='Ensembl Transcript ID (e.g. ENST00000397163.8)',
  )
  parser_gtf.add_argument(
      '--variant',
      help=(
          'Variant string in 1-based chr:pos:ref>alt format (e.g.'
          ' chr15:42387805:C>G)'
      ),
  )
  parser_gtf.add_argument(
      '--region',
      '-r',
      help=(
          'Genomic interval in 1-based closed chr:start-end format (e.g.'
          ' chr15:42380000-42420000)'
      ),
  )
  parser_gtf.add_argument(
      '--all_transcripts',
      action='store_true',
      default=False,
      help='Include all transcripts instead of filtering to MANE Select',
  )
  parser_gtf.add_argument(
      '--protein_coding',
      action='store_true',
      default=False,
      help='Filter to protein-coding genes/transcripts',
  )
  parser_gtf.add_argument(
      '--longest',
      action='store_true',
      default=False,
      help='Filter to longest transcript per gene',
  )
  parser_gtf.add_argument(
      '--exons',
      action='store_true',
      default=False,
      help=(
          'Display detailed 1-based exon boundaries and donor/acceptor'
          ' coordinates'
      ),
  )
  parser_gtf.add_argument(
      '--junctions',
      action='store_true',
      default=False,
      help=(
          'Display canonical and exon-skipping splice junctions with 1-based'
          ' coordinates and constructed Atlas ScoreIds'
      ),
  )
  parser_gtf.add_argument(
      '--track',
      type=str,
      default=None,
      help=(
          'Track name for constructing splice junction ScoreIds (e.g.'
          ' junction_UBERON:0011907 gtex Muscle_Skeletal polyA plus RNA-seq)'
      ),
  )
  parser_gtf.add_argument(
      '--cds',
      action='store_true',
      default=False,
      help='Display 1-based coding sequence (CDS) intervals',
  )
  parser_gtf.add_argument(
      '--utr',
      action='store_true',
      default=False,
      help="Display 1-based 5' and 3' UTR intervals",
  )
  parser_gtf.add_argument(
      '--format',
      choices=['table', 'json', 'tsv'],
      default='table',
      help='Output format',
  )
  parser_gtf.add_argument(
      '--output',
      '-o',
      help='Optional path to export tabular records (TSV, CSV, Parquet, JSON)',
  )

  args = parser.parse_args()

  if args.subcommand == 'annotate':
    handle_annotate(args)
  elif args.subcommand == 'query':
    handle_query(args)
  elif args.subcommand == 'region':
    handle_region(args)
  elif args.subcommand == 'metadata':
    handle_metadata(args)
  elif args.subcommand == 'gtf':
    handle_gtf(args)


if __name__ == '__main__':
  main()
