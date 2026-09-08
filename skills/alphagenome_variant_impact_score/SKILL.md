---
name: alphagenome-variant-impact-score
description: >-
  Score, annotate, and analyze the functional impact of genetic variants using AlphaGenome Variant Impact
  (AVI) scores. Query variants in chr:pos:ref>alt format, annotate VCF/tabular callsets, perform
  saturation mutagenesis window scans (1-based closed chr:start-end), and extract GENCODE v46 GTF
  gene/exon/junction coordinates all via the AlphaGenome Atlas API.
---

# AlphaGenome Variant Impact (AVI) Analysis

Score and prioritize genetic variants using AlphaGenome Variant Impact (AVI)
models via `scripts/alphagenome_atlas_avi.py`.

> [!IMPORTANT] **Research Use Only & Clinical Safety Rules**: The AlphaGenome
> AVI Skill and the underlying AlphaGenome model/Atlas are **strictly research
> tools**. Access to outputs requires an AlphaGenome API key subject to terms of
> service prohibiting clinical use.
>
> 1.  **No Medical Advice or Clinical Diagnosis**: You **MUST NOT** provide
>     medical advice, clinical diagnoses, disease management strategies, or
>     treatment recommendations based on outputs from this skill or the
>     AlphaGenome Atlas.
> 2.  **Strict Molecular & Functional Framing**: A high AVI score reflects
>     **predicted molecular/functional impact** (e.g., disruption of splicing,
>     alteration of transcription factor binding, chromatin accessibility
>     changes, or coding consequences). Frame all findings in terms of molecular
>     mechanisms and biological annotations—never as clinical diagnoses or
>     medical conclusions.
> 3.  **No Diagnostic Leaps**: Never extrapolate high functional impact to
>     clinical disease causation, penetrance, or patient prognosis. If a user
>     asks a clinical or diagnostic question, explicitly clarify that
>     AlphaGenome is a research tool and restrict your answer to the predicted
>     molecular and functional effects.

> [!IMPORTANT] **Always Use AlphaGenome GENCODE v46 GTF
> (`scripts/alphagenome_atlas_avi.py gtf`) for Gene Annotations**: When
> retrieving gene models, transcript IDs, exon coordinates, CDS/UTR regions, or
> splice junction donor/acceptor boundaries, **always use the built-in
> `scripts/alphagenome_atlas_avi.py gtf` command**. Do **NOT** query external
> sources (e.g., Ensembl REST API, UCSC, external GTF databases, or NCBI) for
> gene annotations or transcript coordinates. This ensures that the annotations
> match the scores and website.

--------------------------------------------------------------------------------

## Prerequisites

Run `scripts/alphagenome_atlas_avi.py` using `uv run`:

```bash
# Display CLI help:
uv run scripts/alphagenome_atlas_avi.py --help
```

> [!TIP] **Agent & Programmatic Execution Format**: When invoking the CLI in
> agent workflows, prefer `--format json` or direct file export (`-o <file>`)
> for deterministic, structured parsing rather than extracting fields from
> stdout markdown tables.

### 1. Query Variants (`query`)

Query single or multiple variants in 1-based `chr:pos:ref>alt` format to inspect
scores and the 18 biological feature attribution weights:

```bash
# Query single variant with basic feature importances (stdout JSON preview):
uv run scripts/alphagenome_atlas_avi.py query "chr9:128225994:G>A" --format json

# Query variant with exact underlying Atlas track indices, biosamples, and target genes:
uv run scripts/alphagenome_atlas_avi.py query "chr9:128225994:G>A" --include_track_info --format json

# Query multiple variants and export full track info to a file (prevents stdout overflow):
uv run scripts/alphagenome_atlas_avi.py query "chr9:128225994:G>A" "chr22:36201698:A>C" \
  --include_track_info --format json -o query_results.json
```

*   **Output Size & Redirection**: Single variant table/JSON queries are compact
    (~1.0–1.5 KB). When querying $>3$ variants or passing `--include_track_info
    --format json` (which generates ~3.9 KB per variant), **always export
    directly to a file** using `-o <file.json>` or `-o <file.tsv>` to avoid
    exceeding context window limits.
*   **Typical Runtime**: ~0.5–1.0 s per variant (~12–15 s total including
    environment startup). Single calls with $>100$ variants take $>1$ minute.

### 2. Annotate & Rank Variant Files (`annotate`)

Annotate a VCF (or CSV/TSV/Parquet) in standard Ensembl VEP `CSQ` format:

```bash
uv run scripts/alphagenome_atlas_avi.py annotate \
  --input test_data/example_variants.vcf \
  --output annotated_variants.vcf \
  --top_k 20 \
  --min_phred 15.0 \
  --top_output top_variants.json
```

*   **Output Size & Redirection**: `annotate` streams the full callset directly
    to disk via `--output` (`.vcf`, `.vcf.gz`, `.parquet`, `.tsv`, `.csv`).
    Stdout displays a bounded summary (top candidate table + top 3 modality
    breakdowns, ~4.0–5.5 KB). Use `--top_output <file.json>` when downstream
    tools need machine-readable top candidate data.
*   **Typical Runtime & Callset Scaling**: Throughput is ~10 variants/s
    (default) and ~5 variants/s (with track info). Small callsets ($\le 100$
    variants) take ~15 s. Callsets $\ge 500$ variants execute silently for $>1$
    minute (e.g., 1,000 variants take ~2–3 min; 10,000 variants take ~20 min).

### 3. Saturation Mutagenesis Window Scan (`region`)

Scan a 1-based closed genomic window (`chr:start-end`) to score all possible
single nucleotide substitutions ($3 \times N$ variants for an $N$-bp window):

```bash
uv run scripts/alphagenome_atlas_avi.py region \
  --region chr9:128225990-128226000 \
  --min_phred 15.0 \
  --top_k 20 \
  --output region_hotspots.tsv
```

*   **Output Size & Redirection**: Scanning a 100 bp window produces 300 SNVs
    (~42 KB TSV / ~105 KB JSON), while a 1,000 bp window produces 3,000 SNVs
    (~421 KB TSV / ~1.06 MB JSON). **Always specify `--output
    <file.tsv|parquet>`** to save the complete dataset; stdout will only show a
    top-20 candidate preview.
*   **Typical Runtime**: Queries precomputed dense scores over gRPC in **1.5–3.0
    s** for up to 1,000 bp (default `--max_window_size`).

### 4. Inspect Atlas Database Metadata (`metadata`)

Dump or search registered scorers, the 18 biological feature definitions, or
experimental track catalogs:

```bash
# List all registered scorers:
uv run scripts/alphagenome_atlas_avi.py metadata --scorers --format json

# List the 18 biological feature attribution modalities:
uv run scripts/alphagenome_atlas_avi.py metadata --features --format json

# Search experimental tracks by query keyword with a controlled preview (top 20):
uv run scripts/alphagenome_atlas_avi.py metadata --tracks --query "GATA1" --top_n 20

# Dump complete 9,440-track catalog to Parquet or TSV for offline search:
uv run scripts/alphagenome_atlas_avi.py metadata --tracks --output atlas_tracks.parquet
```

*   **Output Size & Redirection**: `--features` (~1.3 KB) and `--scorers` table
    (~1.7 KB) are safely under 4 KB. The experimental track catalog contains
    **9,440 tracks** (~972 KB table / ~2.55 MB JSON). **Never dump unfiltered
    tracks to stdout**; always supply a narrow `--query`, specify `--top_n 20`,
    or export to `--output atlas_tracks.parquet`.
*   **Typical Runtime**: ~1.5–2.5 s.

### 5. Inspect Gene Structure & Splice Junctions (`gtf`)

Query GENCODE v46 gene annotations, extract 1-based exon boundaries with
donor/acceptor coordinates, and compute canonical and exon-skipping splice
junction coordinates:

```bash
# Query MANE Select exon boundaries for a gene (stdout JSON preview):
uv run scripts/alphagenome_atlas_avi.py gtf --gene CAPN3 --exons --format json

# Query canonical and exon-skipping splice junctions for a variant locus:
uv run scripts/alphagenome_atlas_avi.py gtf --variant "chr15:42387805:C>G" --junctions --format json

# Extract CDS and UTR segments for an Ensembl transcript ID:
uv run scripts/alphagenome_atlas_avi.py gtf --transcript_id ENST00000397163.8 --cds --utr --format json
```

*   **Output Size & Redirection**: Single-gene MANE Select queries with
    `--exons` are compact
    (<2.5 KB). **Always export to `--output <file.tsv|parquet>`or redirect (`>
    gtf_out.json`)** when passing `--all_transcripts`(30–500
    KB),`--region`($>50\text{ kb}$),`--format json`, or querying genes with
    $>30$ exons (e.g.`TTN`, `DMD`).
*   **Typical Runtime & Memory**: Ingests the 318 MB GENCODE v46 feather dataset
    (~4.37 GB RAM). Cached queries execute in **~4–6 s**. Cold-start runs
    downloading the file from GCS take **~5–10 s** on corp/Cloudtop networks,
    but can take up to **60–90 s** on external networks. Do not kill the process
    prematurely during initial download.

--------------------------------------------------------------------------------

## Programmatic Python SDK Usage

When querying AlphaGenome programmatically in Python:

1.  **Client Factory**: Always instantiate the client using
    `atlas.create(api_key)` — do **NOT** instantiate `atlas.AtlasClient()`
    directly (which requires an internal gRPC stub).
2.  **Coordinate System**: `genome.Interval` operates with **0-based half-open
    indexing** (`[start, end)`). When querying a 1-based closed interval
    `chr:start_1_based-end_1_based`, pass `start = start_1_based - 1` and `end =
    end_1_based`.
3.  **Single Variants**: `genome.Variant.from_str("chr:pos:ref>alt")` expects
    **1-based** position coordinates.

```python
import os
from alphagenome.atlas import atlas
from alphagenome.data import genome
import dotenv

dotenv.load_dotenv(os.path.expanduser('~/.env'))
client = atlas.create(os.environ['ALPHAGENOME_API_KEY'])

# Query 1-based closed interval chr11:5225727-5226575 using 0-based half-open [5225726, 5226575)
interval = genome.Interval(chromosome='chr11', start=5225726, end=5226575)
results = client.query_interval(
    interval,
    requested_scorers=['AVI_SCORE', 'AVI_SCORE_FEATURE_IMPORTANCE'],
)

# Query single variant with 1-based coordinate
variant = genome.Variant.from_str('chr11:5225488:A>T')
variant_scores = client.query_variant(
    variant,
    requested_scorers=['AVI_SCORE', 'AVI_SCORE_FEATURE_IMPORTANCE'],
)
```

--------------------------------------------------------------------------------

## Output Schemas & Response Structures

### 1. VCF CSQ & INFO Tag Schema (`annotate`)

When writing annotated VCF files, the following annotations are injected:

*   **Ensembl VEP `CSQ` Format String**:
    `Allele|AVI_PHRED|AVI_RAW|AVI_QUANTILE|AVI_TOP_PERCENTILE|AVI_TOP_FEATURE`
*   **Subfield Definitions**:
    *   `Allele` (*string*): Alternate allele base(s) (e.g. `A`).
    *   `AVI_PHRED` (*float*): Calibrated Phred impact score ($\text{Phred} =
        -10 \cdot \log_{10}(1.0 - \text{quantile})$). Range: `[0.0, ~70.0]`.
        Higher = greater functional impact.
    *   `AVI_RAW` (*float*): Raw continuous model prediction.
    *   `AVI_QUANTILE` (*float*): Calibrated tail quantile ($1 - \text{CDF}$).
        Range: `(0.0, 1.0]`.
    *   `AVI_TOP_PERCENTILE` (*float*): Top percentile of genome-wide SNVs
        ($10^{-\text{Phred}/10} \times 100\%$, e.g. `0.0380` for Top 0.038%).
    *   `AVI_TOP_FEATURE` (*string*): Display name of the top contributing
        biological modality (e.g. `Splicing`, `AlphaMissense`, `ChIP-TF`,
        `DNASE-seq`, `Cactus`).
*   **Standalone INFO Tags** (unless `--vep_csq_only`): `AVI_PHRED=Float`,
    `AVI_RAW=Float`, `AVI_QUANTILE=Float`, `AVI_TOP_PERCENTILE=Float`,
    `AVI_TOP_FEATURE=String`.

### 2. Tabular & JSON Record Schema (`query`, `region`, `--top_output`)

Output records exported to JSON, TSV, CSV, or Parquet contain the following
fields:

*   `rank` (*integer*, present in `--top_output` and `region` tabular exports):
    1-based candidate rank sorted by Phred descending.
*   `variant` (*string*): Genomic variant string in `chr:pos_1_based:ref>alt`
    format (e.g. `chr9:128225994:G>A`, where coordinate is 1-based).
*   `chromosome` (*string*): Contig name with `chr` prefix (e.g. `chr9`).
*   `position` (*integer*): 1-based genomic coordinate.
*   `ref` (*string*): Reference allele base(s).
*   `alt` (*string*): Alternate allele base(s).
*   `avi_phred` (*float*): Calibrated Phred-scaled score.
*   `avi_raw` (*float*): Raw continuous model score.
*   `avi_quantile` (*float*): Tail quantile ($1 - \text{CDF}$).
*   `top_percentile` (*float*): Exact top percentile value (e.g. `0.3421` for
    Top 0.34%).
*   `top_modality` (*string*): Display name of the top contributing biological
    modality (e.g. `Splicing`, `AlphaMissense`, `ChIP-TF`).
*   `top_feature_importance` (*float*): Attribution weight (SHAP value) of the
    top modality.
*   **Optional Attribution Weights (`--include_features`)**: 18 columns
    `fi_<MODALITY>` (e.g. `fi_MERGED_SPLICING`, `fi_ALPHAMISSENSE`,
    `fi_MAX_ABS_RNA_SEQ`, `fi_CACTUS_241_WAY`).
*   **Optional Track Provenance (`--include_track_info`)**:
    `track_idx_<MODALITY>`, `track_name_<MODALITY>`,
    `track_biosample_<MODALITY>`, `track_gene_<MODALITY>`.

### 3. GTF Query Schema (`gtf`)

When querying gene structure with `gtf --format json`, each transcript object
provides:

*   **Gene & Transcript Metadata**: `gene_name`, `gene_id`, `transcript_id`,
    `transcript_type`, `chromosome`, `start`, `end`, `strand`, `is_mane_select`,
    `num_exons` (coordinates are 1-based closed).
*   **Exons Array (`--exons`)**:
    *   `exon_number` (*int*): 1-based exon index in 5' to 3' transcript order.
    *   `start`, `end`, `width` (*int*): 1-based closed exon coordinates and
        length in bp.
    *   `acceptor`, `donor` (*int*): 1-based 5' splice acceptor and 3' splice
        donor coordinates.
    *   `overlaps_variant` (*bool*): True if overlapping the query mutation
        site.
*   **Junctions Array (`--junctions`)**:
    *   `type` (*string*): `Canonical Intron {i}` or `Exon {k} Skipping`.
    *   `upstream_exon`, `downstream_exon` (*int*): 1-based flanking exon
        numbers in 5' to 3' transcript order.
    *   `junction_start`, `junction_end` (*int*): 1-based genomic donor and
        acceptor coordinates.
    *   `intron_length` (*int*): Intron length in bp.
    *   `score_id` (*string*): Constructed Atlas track comparison token.
    *   `atlas_url` (*string*): Clickable deep-link to Atlas track predictions.
*   **CDS & UTR Arrays (`--cds`, `--utr`)**: 1-based closed `start`, `end`,
    `width` (*int*).

--------------------------------------------------------------------------------

## Score Metrics & Interpretation

*   **AVI Phred**: Calibrated score (`Phred = -10 * log10(1.0 - quantile)`):
    *   `Phred >= 40`: **Top 0.01%** predicted impact of all genome-wide SNVs.
    *   `Phred >= 30`: **Top 0.10%** predicted impact of all genome-wide SNVs.
    *   `Phred >= 20`: **Top 1.0%** predicted impact of all genome-wide SNVs.
    *   `Phred >= 15`: **Top 3.16%** predicted impact of all genome-wide SNVs.
    *   `Phred >= 10`: **Top 10.0%** predicted impact of all genome-wide SNVs.
    *   `Phred < 10`: **Bottom 90%** of all genome-wide SNVs.
*   **Top Percentile**: Exact percentage of genome-wide SNVs with equal or
    greater impact: `Top Percentile = 10^(-Phred/10) * 100%` (e.g. Phred 29.7
    $\implies$ Top 0.11%).
*   **Top Modality**: Leading biological feature attribution (`metadata
    --features`).

--------------------------------------------------------------------------------

## Visualizations & Atlas Deep-Linking

> [!IMPORTANT] **Mandatory Atlas Deep-Linking with Variant Scores**: Whenever
> reporting, discussing, or displaying an AVI score for a variant (whether for a
> single variant query, a ranked candidate table, or a genomic region scan), you
> **MUST always provide a clickable deep-link to the
> [AlphaGenome Atlas](https://deepmind.google.com/science/alphagenome/atlas)**
> for each variant.

1.  **User Context Prioritization**: Always prioritize **context from the user
    first** (e.g., disease, tissue, or relevant cell types). When absent, use
    `--include_track_info` with `query` to discover the driving biosample, cell
    line, ontology CURIE, and track index.
2.  **Mandatory Splicing & RNA-seq Co-Plotting Rule**: Whenever plotting or
    visualizing splicing variants, **always plot continuous RNA-seq expression
    alongside splicing tracks** (`SPLICE_JUNCTIONS`, `SPLICE_SITE_USAGE`,
    `SPLICE_SITES`) to evaluate both structural splice disruption and resulting
    transcript abundance changes.
3.  **Atlas Link Construction**: Use the **`alphagenome-atlas-website-links`
    skill** (`scripts/alphagenome_atlas_links.py`) for all URL generation,
    layout configuration (`lItems`), biosample filtering (`f`), and
    `/atlas/track-predictions` comparison chart links.
