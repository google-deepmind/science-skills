---
name: alphagenome-atlas-website-links
description: >-
  Constructs deep-links and URLs for the AlphaGenome Atlas website. Supports generating single-variant exploration
  links (1-based chr:pos:ref>alt), genomic locus views (1-based closed chr:start-end), candidate summary tables,
  and AlphaGenome reference vs. alternate predictions. Use whenever visualizing, exploring, charting, or linking
  genetic variants and genomic loci on the AlphaGenome Atlas, or when asked to inspect, view, or link predictions for a
  genomic variant.
---

# AlphaGenome Atlas Deep-Linking & URL Configuration

Construct and validate deep-links for the AlphaGenome Atlas web application
(`https://deepmind.google.com/science/alphagenome/atlas`).

Base URL: `https://deepmind.google.com/science/alphagenome/atlas`

> [!IMPORTANT] **Mandatory Atlas Deep-Linking with Variant Scores**: Whenever
> presenting, discussing, or scoring genetic variants, you **MUST always provide
> clickable deep-links to the
> [AlphaGenome Atlas](https://deepmind.google.com/science/alphagenome/atlas)**.
> Use `scripts/alphagenome_atlas_links.py` to automate link and table
> generation.

--------------------------------------------------------------------------------

## Prerequisites

```bash
# 1. Single Variant Exploration Link:
uv run scripts/alphagenome_atlas_links.py variant "chr9:128225994:G>A" \
  --biosample K562 \
  --modalities RNA_SEQ,DNASE,CHIP_TF

# 2. Genomic Locus / Interval Link:
uv run scripts/alphagenome_atlas_links.py locus "chr11:5288500-5290500" \
  --biosample K562 \
  --modalities RNA_SEQ,DNASE,CHIP_TF

# 3. Format Candidate Variant Records Table (with embedded clickable links):
uv run scripts/alphagenome_atlas_links.py table --input top_variants.json --biosample K562

# 4. Construct Ref vs. Alt Track Predictions Link (/atlas/track-predictions):
uv run scripts/alphagenome_atlas_links.py track-predictions \
  --variant "chr15:42387805:C>G" \
  --gene CAPN3 \
  --biosample "Muscle_Skeletal"
```

--------------------------------------------------------------------------------

## 2. URL Query Parameters

*   **`q`** (*string*, **Required**): Primary search target. Supports 1-based
    closed intervals (`chr11:5288500-5290500`), gene symbols (`BRCA1`), Ensembl
    IDs (`ENSG00000012048`), or 1-based variants (`chr7:27170000:A>G`).
*   **`m`** (*enum*, Optional): View mode. Defaults to `entity` for
    genes/variants and `locus` for coordinate intervals. Use `variant` for
    variant queries. (Allowed: `locus`, `entity`, `variant`, `motifs`).
*   **`i`** (*string*, Optional): Centered viewport zoom interval in 1-based
    closed `chr:start-end` format (e.g. `chr11:5289310-5289690`). Required for
    automatic motif rendering.
*   **`f`** (*string*, Optional): Comma-separated filter predicates in
    `KEY:VALUE` format (e.g.
    `BIOSAMPLE_NAME:K562,SCORER_MODALITY:RNA-seq,ASSAY_TRANSCRIPTOR_FACTOR:GATA1`).
    Controls visible heatmap rows.
*   **`lItems`** (*string*, Optional): Layout item sequence, AVI score track
    toggle (`avi`), section heatmaps, and pinned tracks list (e.g.
    `avi,section:RNA_SEQ,section:DNASE,pinned:<TrackKey>`).
*   **`scores`** (*string*, Optional): Comma-separated list of `ScoreId` tokens
    for the `/atlas/track-predictions` page comparison (e.g.
    `<ScoreId1>,<ScoreId2>`).
*   **`md`** (*enum*, Optional): Active modality tab selector on the track
    predictions view (`RNA_SEQ`, `SPLICE_JUNCTIONS`, `SPLICE_SITE_USAGE`,
    `DNASE`).
*   **`tpRenames`** (*string*, Optional): Custom title overrides for specific
    score predictions (`ScoreId:CustomTitle`).
*   **`tpLegendTitle`** (*string*, Optional): Custom legend title for the track
    predictions chart card (e.g. `Predicted Gene Expression`).

> [!IMPORTANT] **Variant Query Format**: Variants in `q` must strictly use
> `chr:pos_1_based:ref>alt` format (e.g. `chr7:27170000:A>G` or URL-encoded
> `chr7:27170000:A%3EG`, where the position is 1-based). **Do not use**
> colon-separated alleles (`A:G`) or dbSNP rsIDs (rsIDs are unsupported).

--------------------------------------------------------------------------------

## 3. Multi-Modality Filtering & The Assay Group Gotcha (`f`)

### Filter Groups & Boolean Evaluation

Filters in `f` map to three primary evaluation groups:

*   **`Biosample` Group (`BIOSAMPLE_NAME`, `BIOSAMPLE_TYPE`)**: Evaluated with
    **AND** logic.
*   **`Assay` Group (`SCORER_MODALITY`, `ASSAY_TRANSCRIPTOR_FACTOR`,
    `ASSAY_HISTONE_MARK`)**: Evaluated with **OR** logic.
*   **`Gene` Group (`GENE_NAME`)**: Evaluated with **OR** logic.

### ⚠️ Mandatory Multi-Modality Filter Rule

`RNA-seq` and `DNase` tracks have no transcription factor code
(`transcriptionFactorCode === ""`). If `f` contains *only*
`ASSAY_TRANSCRIPTOR_FACTOR` filters under the Assay group, `RNA-seq` and `DNase`
tracks fail the Assay evaluation and are hidden from the heatmap.

To display `RNA-seq` and `DNase` tracks alongside specific ChIP-seq
transcription factors, **explicitly include `SCORER_MODALITY:RNA-seq` and
`SCORER_MODALITY:DNase`** in `f` (handled automatically by
`scripts/alphagenome_atlas_links.py`):

```
f=BIOSAMPLE_NAME:<CellLine>,SCORER_MODALITY:RNA-seq,SCORER_MODALITY:DNase,ASSAY_TRANSCRIPTOR_FACTOR:<TF1>,ASSAY_TRANSCRIPTOR_FACTOR:<TF2>
```

--------------------------------------------------------------------------------

## 4. Layout Configuration, AVI Scores, & Pinned Tracks (`lItems`)

### Plotting AVI Scores and Modality Sections

*   **AVI Variant Impact Track (`avi`)**: Including `avi` in `lItems` renders
    the top-level AlphaGenome Variant Impact score track for the interval or
    variant.
*   **Database Modality Sections (`section:<MODALITY>`)**: Sections render full
    unpinned heatmaps across all matching tracks for that modality (e.g.
    `section:RNA_SEQ`, `section:DNASE`, `section:CHIP_TF`, `section:ATAC`,
    `section:CAGE`).

### Pinned Tracks & Motif Instances

> [!NOTE] **Track-Specific Motif Guideline**: Pinned Active-ISM tracks with
> motif instances and Contribution Weight Matrix (CWM) logos should **only be
> added when specifically requested for individual tracks**. Only a limited
> subset of tracks (such as key ChIP-TF or RNA-seq tracks relevant to the locus)
> support and benefit from pinned motif overlays. For standard exploration
> links, default section heatmaps
> (`avi,section:RNA_SEQ,section:DNASE,section:CHIP_TF`) without pinned tracks
> are preferred.

Motif instances and CWM logos render **exclusively on pinned tracks** at
base-pair resolution. General section heatmaps do not trigger motif footprint
rendering.

### Pinned Track Key Schema

```
pinned:<TrackMetadataName>:<StrandNumber>:<ScorerShortName>:heatmap:HEATMAP_TILESET_SOURCE_ACTIVE_ISM_SCORES:<TilesetId>
```

*   `<TrackMetadataName>`: Exact track name from production metadata proto,
    URL-encoded (`%20` for spaces).
*   `<StrandNumber>`: `1` (`STRAND_POSITIVE`), `2` (`STRAND_NEGATIVE`), `3`
    (`STRAND_UNSTRANDED`).
*   `<ScorerShortName>`: `RNA_SEQ`, `CHIP_TF`, `DNASE`, `ATAC`, `CAGE`,
    `PROCAP`, `CHIP_HISTONE`.
*   `HEATMAP_TILESET_SOURCE_ACTIVE_ISM_SCORES`: Required source identifier for
    Active-ISM motif layers.
*   `<TilesetId>`: Server-assigned tileset identifier (`17354278441953531756`
    for current production).

### Recipe for Automatic Motif Display on Load

1.  Append `pinned:<PinnedKey>` entries to `lItems` for the specific target
    tracks only.
2.  Set viewport interval `i` to base-pair resolution ($\le 1\text{ bp/px}$,
    window $\le 380\text{ bp}$).
3.  Configure `f` with cell line and transcription factors.

--------------------------------------------------------------------------------

## 5. Track Predictions & Ref vs. Alt Comparisons (`/atlas/track-predictions`)

The dedicated `/atlas/track-predictions` page compares predicted functional
profiles between the Reference and Alternate alleles for selected scores across
genomic windows:

*   **Route**:
    `https://deepmind.google.com/science/alphagenome/atlas/track-predictions`
*   **Visualizations**: Expanded line plots (expression, chromatin
    accessibility, TF binding) and Sashimi arc charts (splice junctions).

### Automated Prediction Link Generation (`scripts/alphagenome_atlas_links.py track-predictions`)

Always construct track prediction URLs using `scripts/alphagenome_atlas_links.py
track-predictions`. Manual `ScoreId` string formatting is error-prone due to
donor/acceptor skipping coordinates, strand orientation (+/-), and genic vs.
non-genic suffix rules. The script automatically handles coordinate extraction
from GENCODE v46, track catalog resolution, and URL synthesis.

```bash
# Variant & Gene:
uv run scripts/alphagenome_atlas_links.py track-predictions \
  --variant "chr15:42387805:C>G" \
  --gene CAPN3 \
  --biosample "Muscle_Skeletal" \
  --modalities SPLICE_JUNCTIONS,RNA_SEQ,DNASE,CHIP_TF \
  --tf CTCF

# Interval/Locus query:
uv run scripts/alphagenome_atlas_links.py track-predictions \
  --variant "chr15:42387805:C>G" \
  --interval "chr15:41869312-42917888" \
  --biosample "Muscle_Skeletal" \
  --modalities SPLICE_JUNCTIONS,RNA_SEQ,DNASE,CHIP_TF
```

### Supported CLI Options for `track-predictions`

*   **`--variant`, `-v`** (*string*, default: `None`): Variant string in
    `chr:pos_1_based:ref>alt` format.
*   **`--gene`, `-g`** (*string*, default: `None`): Target gene symbol (bounds
    `i=` viewport and computes splice junctions).
*   **`--gene_id`** (*string*, default: `None`): Target Ensembl gene ID (e.g.
    `ENSG00000092529.26`).
*   **`--interval`, `-i`** (*string*, default: `None`): Genomic interval
    viewport in `chr:start-end` format.
*   **`--biosample`, `-b`** (*string*, default: `Muscle_Skeletal`): Target
    biosample or tissue query (e.g. `Muscle_Skeletal`, `K562`, `Whole_Blood`).
*   **`--modalities`, `-m`** (*string*, default:
    `SPLICE_JUNCTIONS,RNA_SEQ,DNASE,CHIP_TF`): Comma-separated list of
    modalities (`SPLICE_JUNCTIONS`, `RNA_SEQ`, `DNASE`, `ATAC`, `CHIP_TF`).
*   **`--tf`** (*string*, default: `CTCF`): Transcription factor name for
    ChIP-TF tracks (e.g. `CTCF`, `GATA1`).
*   **`--rename`** (*string*, default: `None`): Custom track rename overrides in
    the chart card.
*   **`--legend_title`** (*string*, default: `None`): Custom legend header for
    the chart card.
*   **`--format`** (*enum*, default: `table`): Output format (`table`, `url`,
    `json`).

> [!IMPORTANT] **Mandatory Splicing & RNA-seq Co-Plotting Rule**: When
> generating `/atlas/track-predictions` deep-links, plotting, or visualizing
> variant impact data for splicing variants, **always plot continuous RNA-seq
> expression alongside splicing tracks** (`SPLICE_JUNCTIONS`,
> `SPLICE_SITE_USAGE`, `SPLICE_SITES`). Splicing mutations frequently activate
> cryptic splice junctions and trigger nonsense-mediated decay (NMD) or alter
> total transcript output; assessing splice junctions (sashimi arcs) together
> with continuous RNA-seq read coverage is required to observe both the
> structural splice defect and the resulting change in overall transcript
> abundance.

> [!IMPORTANT] **Always Provide Bounded `i=` in Track Prediction URLs**:
> Omitting `scores=` or leaving the genomic interval (`i=`) unbounded causes the
> web application to attempt querying all matching tracks across the broader
> locus, leading to severe latency or page hanging. `alphagenome_atlas_links.py
> track-predictions` automatically bounds `i=` to the target gene or requested
> interval.
