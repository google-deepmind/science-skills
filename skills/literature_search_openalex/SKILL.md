---
name: literature-search-openalex
description: >
  Query the OpenAlex scholarly database for research papers, authors,
  institutions, topics, sources, publishers, funders, geo-locations, and
  keywords. Use when searching academic papers, resolving DOIs, downloading
  open-access PDFs, finding an author's publications, aggregating bibliometric
  data (citation counts, h-index, impact factor), exploring the research
  taxonomies, or performing DOI lookups.
---

# OpenAlex Skill

## Prerequisites

1.  **`uv`**: Read the `uv` skill and follow its Setup instructions to ensure
    `uv` is installed and on PATH.
2.  **User Notification**: If .licenses/literature_search_openalex_LICENSE.txt
    does not already exist in the workspace root directory then (1) prominently
    notify the user to check the terms at https://help.openalex.org/ and to
    always check the license of the papers retrieved by the skill for any
    restrictions, then (2) create the file recording the notification text and
    timestamp.
3.  **`.env` file**: Make sure the `.env` file exists in your home directory.
    Create one if it does not exist.
4.  **`OPENALEX_API_KEY`** (optional but recommended): The skill works without a
    key, but keyless requests share a $0.10/day budget per IP address. A free
    key has its own $1/day budget and higher rate limits, and is required to
    download full text from content.openalex.org (`download-pdf`). Adding
    `mailto` (the former "polite pool") no longer raises limits. You can obtain
    a free key at OpenAlex.org → account settings. Whenever checking for or
    requesting this key, you **MUST** use the safe credentials protocol in the
    `credentials` skill, without blocking keyless execution if it is unset.

## Core Rules

1.  **List Sources.** If this skill is used, ensure this is mentioned in the
    output AND list the URLs of all papers that were used in producing the
    output.
2.  **Resolve before filter.** NEVER filter by name. Always `resolve` a name to
    an ID first, then use that ID in `--filter`.
3.  **Use the CLI only.** Never call the API via `curl`/`urllib`. The CLI
    handles retries and rate limiting.
4.  **No fabrication.** Never invent OpenAlex IDs or DOIs. Use `resolve`/`get`
    to look them up. Report empty results accurately.
5.  **API key.** If a command returns 401, returns 429 because of a rate limit
    or exhausted daily budget, or you need high-volume queries, you **MUST** use
    the safe credentials protocol in the `credentials` skill to check for and
    request the `OPENALEX_API_KEY` to help the user add it to their `.env` file.
    A free key does not fix a 429 `Plan upgrade required` error.
6.  **Keep output small.** Always use `--select` and `--per-page 5–10` for
    overview queries. Pipe `filter` output to a file (`> results.json`), then
    slim with `jq` before reading into context.

## Rate Limits

-   **Without key:** 10 req/s; $0.10/day budget per IP address, shared by all
    keyless requests from that IP.
-   **With free key:** 30 req/s; $1/day budget.
-   **Annual plan:** 50 req/s.

Operation              | Cost
---------------------- | -------
Singleton `get`        | Free
`filter`               | $0.0001
`--search` / `resolve` | $0.001
`download-pdf`         | $0.01

## CLI Reference

```
uv run scripts/openalex_cli.py [--api-key KEY] <command> [flags]
```

Entity types (shared across commands): `works`, `authors`, `sources`,
`institutions`, `topics`, `domains`, `fields`, `subfields`, `sdgs`, `countries`,
`continents`, `languages`, `keywords`, `publishers`, `funders`, `work-types`,
`source-types`, `institution-types`, `licenses`

### Commands

**resolve** `<entity> <query>` — Name → ID candidates. Returns `id`,
`display_name`, `hint`. Use `--per-page N` for more candidates.

**get** `<entity> <id>` — Full metadata for one entity. Accepts short ID
(`W2741809807`), full URL, or DOI URL. Use `--select` to limit fields.

**filter** `<entity>` — Search/filter entities. Key flags are:

-   `--search <query>`: Full-text search (10× cost of `--filter`)
-   `--filter <expr>`: Filter expressions. Use `,` for AND and `|` for OR.
-   `--sort <field:dir>`: Sort results (e.g., `cited_by_count:desc`)
-   `--select <fields>`: Limit the fields returned in the output.
-   `--group-by <field>`: Aggregate results by a specific field.
-   `--per-page <N>`: Number of results per page (default 25, max 100).
-   `--page <N>`: Specify the page number to retrieve.
-   `--sample <N>`: Get a random sample of up to 10,000 results.
-   `--seed <N>`: Seed for reproducible sampling.

**download-pdf** `<work-id> <output-path>` — Download PDF
(`content.openalex.org` requires `OPENALEX_API_KEY`; falls back to alternative
`pdf_url` locations from work metadata if unauthenticated or if the primary
server fails). Always run `download-pdf` and verify any downloaded PDF is
non-empty and not corrupted; if `OPENALEX_API_KEY` is missing, also refer to the
`credentials` skill to help the user add it.

**rate-limit** — Check current rate limit and daily credit status (reports
shared per-IP limits when keyless, or key-specific limits when
`OPENALEX_API_KEY` is set; if `OPENALEX_API_KEY` is not set, also refer to the
`credentials` skill to help the user add it).

### Search Tips

-   If `resolve` returns no matches, try alternate spellings or abbreviations.
-   If `--search` returns 0 results, try broader terms (max 3 retries).
-   If `resolve` returns multiple candidates, present them to the user with
    `display_name` and `hint` for manual selection.

## Entity References

Consult `references/` for valid filter, sort, and group-by fields per entity:

-   [Works](references/works.md) — [Authors](references/authors.md) —
    [Sources](references/sources.md)
-   [Institutions](references/institutions.md) — [Topics](references/topics.md)
    — [Taxonomy](references/taxonomy.md)
-   [Geo & Language](references/geo_and_language.md) —
    [Publishers & Funders](references/publishers_funders.md)
-   [Type Values](references/type_values.md)

For anything not covered here, or if the API behaves differently from what this
skill describes, check the official docs index at
https://help.openalex.org/llms.txt.

## Common Workflows

```bash
# Author's works (resolve → filter)
uv run scripts/openalex_cli.py resolve authors "Geoffrey Hinton"
uv run scripts/openalex_cli.py filter works \
  --filter "authorships.author.id:A5108093963" \
  --sort "cited_by_count:desc" --per-page 10 > papers.json
cat papers.json | jq '[.results[] | {id, title: .display_name, year: .publication_year, citations: .cited_by_count}]'

# DOI lookup
uv run scripts/openalex_cli.py get works "https://doi.org/10.1038/s41586-021-03819-2"

# Bulk DOI lookup (up to 100)
uv run scripts/openalex_cli.py filter works \
  --filter "doi:10.1234/a|10.1234/b|10.1234/c" --per-page 100 > results.json

# Institutional impact by year
uv run scripts/openalex_cli.py resolve institutions "MIT"
uv run scripts/openalex_cli.py filter works \
  --filter "authorships.institutions.id:I63966007" \
  --group-by "publication_year" > mit_by_year.json

# Random sample
uv run scripts/openalex_cli.py filter works \
  --filter "publication_year:2023,is_oa:true" \
  --sample 100 --seed 42 > results.json
```

## Error Handling

The CLI already retries 429s with backoff, except `Plan upgrade required` ones,
which fail immediately with an explanation. Don't blindly retry either kind; the
`Server body` in the error output tells them apart.

Code | Meaning                                  | Action
---- | ---------------------------------------- | ------
401  | Unauthorized                             | You MUST use safe credentials protocol in credentials skill to help user add API key to `.env`
404  | Not found                                | Verify ID; try `resolve` first
429  | Body says `Plan upgrade required`        | Paid-plan feature (e.g. a plan-gated filter). Do not retry; a free key won't help. Drop the filter or inform user; see https://openalex.org/pricing
429  | Body says rate limit or budget exhausted | No key: you MUST use safe credentials protocol in credentials skill to help user add API key to `.env`. With key: wait and retry (rate limit) or tell user the daily budget is spent

Known plan-gated filters: `from_created_date`, `from_updated_date`,
`to_updated_date`.

Never fabricate results on empty responses — report accurately and suggest
alternate search terms.
