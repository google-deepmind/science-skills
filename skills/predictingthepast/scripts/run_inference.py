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

"""Run Predicting The Past inference on preprocessed ancient text.

Supports two models: Aeneas (Latin) and Ithaca (Ancient Greek).
Performs up to three tasks per input: geographical and chronological
attribution, text restoration (if gaps marked with ? or # are present),
and contextualization via parallel inscription retrieval. Optionally
generates a text embedding vector.

For texts exceeding 750 characters, the input is split into overlapping
windows and attribution scores are averaged across all windows.
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "jax",
#   "predictingthepast",
# ]
# ///

import argparse
import functools
import json
import os
import pickle
import ssl
import sys
import tempfile
from typing import Any
import urllib.error
import urllib.request

import jax
from predictingthepast.eval import inference
from predictingthepast.models import model as model_lib
from predictingthepast.util import alphabet as util_alphabet

# Per-language model checkpoint, dataset, and retrieval-embedding filenames.
_LATIN_FILES: dict[str, str] = {
    'checkpoint': 'aeneas_117149994_2.pkl',
    'dataset': 'led.json',
    'retrieval': 'led_emb_xid117149994.pkl',
}

_GREEK_FILES: dict[str, str] = {
    'checkpoint': 'ithaca_153143996_2.pkl',
    'dataset': 'iphi.json',
    'retrieval': 'iphi_emb_xid153143996.pkl',
}

_VALID_MODEL_FILENAMES: frozenset[str] = frozenset(
    _LATIN_FILES.values()
) | frozenset(_GREEK_FILES.values())

# Download configuration:
# Chunks are kept well below 32 MiB proxy limits (e.g. Harpoon/Trawler).
_DOWNLOAD_CHUNK_SIZE: int = 16 * 1024 * 1024  # 16 MiB
_DOWNLOAD_TIMEOUT_SECONDS: int = 120  # ~136 kB/s floor before timeout
_DOWNLOAD_PROBE_TIMEOUT_SECONDS: int = 60

# Region-name lookup files mapping numeric location IDs to human-readable names.
# These files are bundled in the skill's references/ directory.
_LATIN_REGION_FILE: str = 'led-region-sub.txt'
_GREEK_REGION_FILE: str = 'iphi-region-sub.txt'

# Path to the skill's references/ directory, resolved relative to this script.
_REFERENCES_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'references'
)

# Date-bin boundaries: years are discretised into 10-year bins from 800 BCE
# to 800 CE, giving 160 bins total.
_DATE_MIN: int = -800
_DATE_MAX: int = 800
_DATE_INTERVAL: int = 10

_MODELS_DIR: str = os.path.join(
    os.path.expanduser('~'), '.predictingthepast', 'models'
)

GCS_BASE: str = 'https://storage.googleapis.com/ithaca-resources/models'

# Windowing constraints: texts longer than _MAX_LEN characters are split into
# overlapping windows.  The final window is discarded if shorter than _MIN_LEN.
_MAX_LEN: int = 750
_MIN_LEN: int = 25


@functools.cache
def _load_region_names(language: str) -> list[str]:
  """Loads region names from a line-delimited text file.

  Args:
    language: 'latin' or 'greek', selects the correct region file.

  Returns:
    Ordered list of region name strings, indexed by location_id.
  """
  filename = _LATIN_REGION_FILE if language == 'latin' else _GREEK_REGION_FILE
  path = os.path.join(_REFERENCES_DIR, filename)
  if not os.path.isfile(path):
    print(f'WARNING: Region file not found: {path}', file=sys.stderr)
    return []
  with open(path, 'r', encoding='utf-8') as f:
    return [line.strip() for line in f if line.strip()]


def _year_bin_to_year(bin_idx: int) -> int:
  """Converts a year-bin index to the midpoint calendar year of that bin."""
  return _DATE_MIN + bin_idx * _DATE_INTERVAL + _DATE_INTERVAL // 2


def _load_checkpoint(
    path: str, language: str
) -> tuple[dict[str, Any], Any, Any, Any, Any]:
  """Loads a model checkpoint from a pickle file.

  Args:
    path: Absolute path to the .pkl checkpoint.
    language: 'latin' or 'greek', determines which alphabet to instantiate.

  Returns:
    A tuple of (model_config, region_map, alphabet, params, forward_fn).
  """
  with open(path, 'rb') as f:
    checkpoint = pickle.load(f)

  params = jax.device_put(checkpoint['params'])
  model = model_lib.Model(**checkpoint['model_config'])
  forward = model.apply
  region_map = checkpoint['region_map']

  if language == 'latin':
    alphabet = util_alphabet.LatinAlphabet()
  elif language == 'greek':
    alphabet = util_alphabet.GreekAlphabet()
  else:
    raise ValueError(f'Unknown language: {language}')

  return checkpoint['model_config'], region_map, alphabet, params, forward


def _enrich_attribution(
    attr: Any | None,
    region_names: list[str],
) -> dict[str, Any] | None:
  """Enriches attribution result with years and region names.

  Calculates the continuous calendar year from date bins, maps region IDs to
  human-readable strings, and adds helper fields for summaries.

  Args:
    attr: The raw attribution result.
    region_names: List of region names indexed by location_id.

  Returns:
    Enriched attribution dictionary, or None if input is None.
  """
  if not attr:
    return None

  if hasattr(attr, 'dict'):
    attr_dict = attr.dict()
  elif hasattr(attr, 'json'):
    attr_dict = json.loads(attr.json())
  else:
    attr_dict = attr

  if 'year_scores' in attr_dict:
    scores = attr_dict['year_scores']
    years = [_year_bin_to_year(i) for i in range(len(scores))]
    attr_dict['year_distribution'] = {str(y): s for y, s in zip(years, scores)}
    if scores:
      top_idx = max(range(len(scores)), key=lambda i: scores[i])
      attr_dict['top_year'] = years[top_idx]
      total = sum(scores)
      if total > 0:
        attr_dict['weighted_average_year'] = round(
            sum(y * s for y, s in zip(years, scores)) / total
        )

  if region_names and 'locations' in attr_dict:
    for loc in attr_dict['locations']:
      loc_id = loc.get('location_id', -1)
      if 0 <= loc_id < len(region_names):
        loc['location_name'] = region_names[loc_id]

  return attr_dict


def _enrich_contextualization(
    ctx: Any | None,
    region_names: list[str],
) -> dict[str, Any] | None:
  """Enriches contextualization result with region names.

  Maps parallel arrays of location IDs to human-readable region names.

  Args:
    ctx: The raw contextualization result.
    region_names: List of region names indexed by location_id.

  Returns:
    Enriched contextualization dictionary, or None if input is None.
  """
  if not ctx:
    return None

  if hasattr(ctx, 'dict'):
    ctx_dict = ctx.dict()
  elif hasattr(ctx, 'json'):
    ctx_dict = json.loads(ctx.json())
  else:
    ctx_dict = ctx

  if region_names and 'location_ids' in ctx_dict:
    ctx_dict['location_names'] = [
        region_names[lid]
        if lid is not None and 0 <= lid < len(region_names)
        else None
        for lid in ctx_dict['location_ids']
    ]
  return ctx_dict


def split_into_windows(
    text: str,
    max_len: int = _MAX_LEN,
    overlap: float = 0.33,
) -> list[str]:
  """Splits text into overlapping windows.

  If text exceeds max_len, it is divided into chunks. An overlap ensures
  context is preserved across boundaries. if the final window is too small,
  it expands backwards to maintain length.

  Args:
    text: Input text string.
    max_len: Maximum length of each window.
    overlap: Overlap fraction between windows.

  Returns:
    List of text windows.
  """
  if len(text) <= max_len:
    return [text]

  step = int(max_len * (1 - overlap))
  windows = []
  for start in range(0, len(text), step):
    window = text[start : start + max_len]
    if len(window) < _MIN_LEN:
      if windows:
        new_start = max(0, len(text) - max_len)
        windows[-1] = text[new_start:]
      break
    windows.append(window)
    if start + max_len >= len(text):
      break
  return windows


def average_attributions(
    attr_list: list[dict[str, Any] | None],
) -> dict[str, Any] | None:
  """Averages attribution scores across windows.

  Takes a list of attribution dictionaries (one per window) and returns a
  single dictionary with numerical scores (like date_scores and region_scores)
  averaged together.

  Args:
    attr_list: List of attribution dictionaries to average.

  Returns:
    A dictionary with averaged scores, or None if input list is empty.
  """
  if not attr_list or not any(attr_list):
    return None

  valid = [a for a in attr_list if a is not None]
  if len(valid) == 1:
    return valid[0]

  averaged = json.loads(json.dumps(valid[0]))

  if 'date_scores' in averaged:
    n = len(valid)
    scores = averaged['date_scores']
    for attr in valid[1:]:
      if 'date_scores' in attr:
        for i, score in enumerate(attr['date_scores']):
          if i < len(scores):
            scores[i] = scores[i] + score
    averaged['date_scores'] = [s / n for s in scores]

  if 'region_scores' in averaged:
    n = len(valid)
    scores = averaged['region_scores']
    for attr in valid[1:]:
      if 'region_scores' in attr:
        for i, score in enumerate(attr['region_scores']):
          if i < len(scores):
            scores[i] = scores[i] + score
    averaged['region_scores'] = [s / n for s in scores]

  return averaged


def _download_model_file(url: str, dest_path: str):
  """Downloads a model file from a URL to a local path using 16 MiB Range chunks."""
  print(f'[*] Downloading {url} to {dest_path}...', file=sys.stderr)

  cafile = os.environ.get('SSL_CERT_FILE') or os.environ.get(
      'REQUESTS_CA_BUNDLE'
  )
  ssl_ctx = ssl.create_default_context(cafile=cafile) if cafile else None

  dest_dir = os.path.dirname(dest_path)
  # Make sure the folder exists before trying to save the file
  os.makedirs(dest_dir, exist_ok=True)

  tmp_path = None
  try:
    # Open the connection to the URL to probe total file size
    probe_req = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
    with urllib.request.urlopen(
        probe_req, timeout=_DOWNLOAD_PROBE_TIMEOUT_SECONDS, context=ssl_ctx
    ) as resp:
      content_range = resp.headers.get('Content-Range', '')
      if '/' in content_range:
        total_size = int(content_range.split('/')[-1])
      else:
        total_size = int(resp.headers.get('Content-Length', 0))

    # Use a named temporary file in the destination directory to ensure
    # atomic replacement on the same filesystem.
    with tempfile.NamedTemporaryFile(
        dir=dest_dir,
        prefix=f'{os.path.basename(dest_path)}.',
        suffix='.tmp',
        delete=False,
    ) as tmp_file:
      tmp_path = tmp_file.name

    # Open the local file and save the downloaded data in chunks
    start = 0
    if total_size > 0:
      with open(tmp_path, 'wb') as f:
        while start < total_size:
          end = min(start + _DOWNLOAD_CHUNK_SIZE - 1, total_size - 1)
          req = urllib.request.Request(
              url, headers={'Range': f'bytes={start}-{end}'}
          )
          with urllib.request.urlopen(
              req, timeout=_DOWNLOAD_TIMEOUT_SECONDS, context=ssl_ctx
          ) as resp:
            data = resp.read()
            if not data:  # Stop when there is no more data
              break
            f.write(data)
            start += len(data)
    else:
      with urllib.request.urlopen(
          url, timeout=_DOWNLOAD_TIMEOUT_SECONDS, context=ssl_ctx
      ) as resp:
        with open(tmp_path, 'wb') as f:
          while True:
            chunk = resp.read(8192)
            if not chunk:  # Stop when there is no more data
              break
            f.write(chunk)
            start += len(chunk)

    if total_size and start != total_size:
      raise OSError(
          f'Incomplete download for {url}: received {start} bytes, expected'
          f' {total_size} bytes'
      )

    os.replace(tmp_path, dest_path)
    tmp_path = None
    print(
        f'[*] Successfully downloaded to {dest_path} ({start} bytes)',
        file=sys.stderr,
    )

  except Exception as e:
    # Clean up the temporary file on error.
    if tmp_path and os.path.exists(tmp_path):
      os.remove(tmp_path)
    raise RuntimeError(
        f'Failed to download model file from {url} to {dest_path}: {e}'
    ) from e


def _load_or_redownload(path: str, url: str, loader_fn: Any) -> Any:
  """Loads a file with loader_fn, re-downloading if truncated or corrupted."""
  try:
    return loader_fn(path)
  except (pickle.UnpicklingError, EOFError, json.JSONDecodeError) as e:
    filename = os.path.basename(path)
    if filename not in _VALID_MODEL_FILENAMES:
      raise RuntimeError(
          f'Refusing to delete unrecognized file {path}: not in'
          f' _VALID_MODEL_FILENAMES ({_VALID_MODEL_FILENAMES})'
      ) from e

    print(
        f'WARNING: Corrupted or truncated file at {path} ({e});'
        ' re-downloading...',
        file=sys.stderr,
    )
    if os.path.exists(path):
      os.remove(path)
    _download_model_file(url, path)
    return loader_fn(path)


def _load_resources(
    args: argparse.Namespace,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
  """Loads the model checkpoint, dataset, and retrieval embeddings.

  Validates that all required files exist under args.models_dir before loading.

  Args:
    args: Parsed command-line arguments.

  Returns:
    A tuple of (model_config, region_map, alphabet, params, forward_fn,
    dataset, retrieval).
  """
  files = _LATIN_FILES if args.language == 'latin' else _GREEK_FILES
  for _, filename in files.items():
    path = os.path.join(args.models_dir, filename)
    if not os.path.isfile(path):
      url = f'{GCS_BASE}/{filename}'
      _download_model_file(url, path)

  checkpoint_path = os.path.join(args.models_dir, files['checkpoint'])
  print(f'Loading checkpoint from {checkpoint_path}...', file=sys.stderr)
  model_config, region_map, alphabet, params, forward = _load_or_redownload(
      checkpoint_path,
      f"{GCS_BASE}/{files['checkpoint']}",
      lambda p: _load_checkpoint(p, args.language),
  )

  dataset_path = os.path.join(args.models_dir, files['dataset'])
  retrieval_path = os.path.join(args.models_dir, files['retrieval'])
  print(f'Loading dataset from {dataset_path}...', file=sys.stderr)
  dataset = _load_or_redownload(
      dataset_path,
      f"{GCS_BASE}/{files['dataset']}",
      inference.load_dataset,
  )
  print(f'Loading retrieval from {retrieval_path}...', file=sys.stderr)
  retrieval = _load_or_redownload(
      retrieval_path,
      f"{GCS_BASE}/{files['retrieval']}",
      inference.load_retrieval,
  )

  return model_config, region_map, alphabet, params, forward, dataset, retrieval


def _process_windows(
    *,
    windows: list[str],
    args: argparse.Namespace,
    forward: Any,
    params: Any,
    alphabet: Any,
    vocab_char_size: int,
    dataset: Any,
    retrieval: Any,
    region_map: Any,
) -> tuple[
    list[dict[str, Any] | None],
    list[dict[str, Any] | None],
    list[dict[str, Any] | None],
]:
  """Runs attribution, restoration, and contextualization on each window.

  Each window is processed independently.  Only the tasks selected via
  --attribute, --restore, and --contextualize flags are executed.

  Args:
    windows: List of text windows produced by split_into_windows().
    args: Parsed command-line arguments.
    forward: JAX model forward function.
    params: JAX model parameters.
    alphabet: Language-specific alphabet encoder.
    vocab_char_size: Character vocabulary size from model_config.
    dataset: Loaded dataset for contextualization.
    retrieval: Loaded retrieval embeddings for contextualization.
    region_map: Region ID mapping from the checkpoint.

  Returns:
    Three parallel lists of per-window results for attribution, restoration,
    and contextualization.  Entries are None on failure or when skipped.
  """
  all_attr = []
  all_restore = []
  all_ctx = []

  include_test = not args.contextualize_exclude_test_valid

  for i, window in enumerate(windows):
    print(f'Processing window {i+1}/{len(windows)}...', file=sys.stderr)

    # 1. Attribution
    if args.attribute:
      try:
        attr = inference.attribute(
            window,
            forward=forward,
            params=params,
            alphabet=alphabet,
            vocab_char_size=vocab_char_size,
        )
        attr_dict = _enrich_attribution(attr, [])
        if attr_dict:
          attr_dict['input_text'] = window
        all_attr.append(attr_dict)
      except Exception as e:
        print(f'Attribution failed for window {i+1}: {e}', file=sys.stderr)
        all_attr.append(None)
    else:  # Attribution not requested.
      all_attr.append(None)

    # 2. Restoration
    if args.restore and ('?' in window or '#' in window):
      try:
        restore = inference.restore(
            window,
            forward=forward,
            params=params,
            alphabet=alphabet,
            vocab_char_size=vocab_char_size,
            beam_width=args.restore_beam_width,
            temperature=args.restore_temperature,
            unk_restoration_max_len=args.restore_max_len,
        )
        if hasattr(restore, 'dict'):
          restore_dict = restore.dict()
        elif hasattr(restore, 'json'):
          restore_dict = json.loads(restore.json())
        else:
          restore_dict = restore

        if restore_dict:
          restore_dict['input_text'] = window
        all_restore.append(restore_dict)
      except Exception as e:
        print(f'Restoration failed for window {i+1}: {e}', file=sys.stderr)
        all_restore.append(None)
    else:
      all_restore.append(None)

    # 3. Contextualization
    if args.contextualize:
      try:
        ctx = inference.contextualize(
            text=window,
            dataset=dataset,
            retrieval=retrieval,
            forward=forward,
            params=params,
            alphabet=alphabet,
            region_map=region_map,
            include_test=include_test,
            top_k=args.contextualize_top_k,
        )
        if hasattr(ctx, 'dict'):
          ctx_dict = ctx.dict()
        elif hasattr(ctx, 'json'):
          ctx_dict = json.loads(ctx.json())
        else:
          ctx_dict = ctx
        all_ctx.append(ctx_dict)
      except Exception as e:
        print(
            f'Contextualization failed for window {i+1}: {e}', file=sys.stderr
        )
        all_ctx.append(None)
    else:  # Contextualization not requested.
      all_ctx.append(None)

  return all_attr, all_restore, all_ctx


def _handle_output(
    *,
    output: dict[str, Any],
    args: argparse.Namespace,
    final_attr: dict[str, Any] | None,
    final_restore: list[dict[str, Any] | None] | None,
    final_ctx: list[dict[str, Any] | None],
    embedding: list[float] | None = None,
) -> None:
  """Prints a human-readable inference summary and writes the output JSON.

  Attribution and restoration summaries are printed to stderr. Contextualization
  results (including full texts) are only written to the output JSON file —
  use jq or a script to extract them.

  Args:
    output: Combined output dictionary to serialise as JSON.
    args: Parsed command-line arguments (used for --output_json path).
    final_attr: Enriched attribution dictionary, or None.
    final_restore: List of per-window restoration dictionaries, or None.
    final_ctx: List of per-window contextualization dictionaries.
    embedding: List of embedding floats, or None if not requested.
  """
  print('\n=== Inference Summary ===', file=sys.stderr)

  if final_attr:
    locations = final_attr.get('locations', [])[:10]
    if locations:
      print('\nTop Locations:', file=sys.stderr)
      for i, loc in enumerate(locations, 1):
        name = loc.get('location_name', 'Unknown')
        score = loc.get('score', 0)
        print(f'  {i:2d}. {name} ({score:.4f})', file=sys.stderr)

    top_year = final_attr.get('top_year')
    avg_year = final_attr.get('weighted_average_year')
    if top_year is not None:
      print(f'\nMost Probable Year: {top_year} CE', file=sys.stderr)
    if avg_year is not None:
      print(f'Weighted Average Year: {avg_year} CE', file=sys.stderr)
  elif args.attribute:
    print('\nAttribution: failed or empty', file=sys.stderr)

  if final_restore:
    print('\nTop Restorations:', file=sys.stderr)
    count = 0
    for window_restore in final_restore:
      if not window_restore:
        continue
      for pred in window_restore.get('predictions', []):
        if count >= 10:
          break
        restored_indices = pred.get('restored', [])
        text = pred.get('text', '')
        restored_text = ''.join(
            text[i] for i in restored_indices if i < len(text)
        )
        score = pred.get('score', 0)
        count += 1
        print(f'  {count:2d}. {restored_text} ({score:.6f})', file=sys.stderr)
      if count >= 10:
        break
  elif args.restore:
    print('\nRestoration: no ? or # markers found', file=sys.stderr)

  if any(c for c in final_ctx if c is not None):
    ctx_count = sum(len(c.get('ids', [])) for c in final_ctx if c is not None)
    print(
        f'\nContextualization: {ctx_count} parallels written to output_json',
        file=sys.stderr,
    )
  elif args.contextualize:
    print('\nContextualization: failed or empty', file=sys.stderr)

  if embedding is not None:
    print(f'\nEmbedding ({len(embedding)} dimensions):', file=sys.stderr)
    print(repr(embedding), file=sys.stderr)

  print('\n=========================\n', file=sys.stderr)

  if args.output_json:
    with open(args.output_json, 'w') as f:
      json.dump(output, f, indent=2)
  else:
    print(json.dumps(output, indent=2))


def generate_embedding(
    *,
    text: str,
    forward: Any,
    params: Any,
    alphabet: Any,
) -> list[float] | None:
  """Generates a text embedding vector.

  Args:
    text: Input text string.
    forward: JAX model forward function.
    params: JAX model parameters.
    alphabet: Language-specific alphabet encoder.

  Returns:
    List of embedding floats, or None on failure.
  """
  try:
    emb = inference._generate_text_emb(
        params, forward, alphabet, text, emb_mode='avg'
    )
    return emb.tolist()
  except Exception as e:
    print(f'Embedding generation failed: {e}', file=sys.stderr)
    return None


def main() -> None:
  """Entry point: parses args, loads models, runs inference, and outputs."""
  args = _parse_args()

  if not (args.attribute or args.contextualize or args.restore):
    print(
        'ERROR: At least one of --attribute, --contextualize, or --restore '
        'must be provided.',
        file=sys.stderr,
    )
    sys.exit(1)

  if args.input:
    text = args.input
  else:
    with open(args.input_file, 'r', encoding='utf-8') as f:
      text = f.read()

  if args.restore and ('?' not in text and '#' not in text):
    print(
        'WARNING: --restore was requested but no ? or # found in input. '
        'Restoration will be skipped.',
        file=sys.stderr,
    )

  windows = split_into_windows(
      text, max_len=_MAX_LEN, overlap=args.window_overlap
  )
  print(
      f'Text length: {len(text)} chars, split into {len(windows)} window(s)',
      file=sys.stderr,
  )

  model_config, region_map, alphabet, params, forward, dataset, retrieval = (
      _load_resources(args)
  )
  vocab_char_size = model_config['vocab_char_size']

  all_attr, all_restore, all_ctx = _process_windows(
      windows=windows,
      args=args,
      forward=forward,
      params=params,
      alphabet=alphabet,
      vocab_char_size=vocab_char_size,
      dataset=dataset,
      retrieval=retrieval,
      region_map=region_map,
  )

  if args.attribute:
    if len(windows) > 1:
      final_attr = average_attributions(all_attr)
    else:
      final_attr = all_attr[0] if all_attr else None
    region_names_list = _load_region_names(args.language)
    if final_attr:
      final_attr = _enrich_attribution(final_attr, region_names_list)
  else:
    final_attr = None
    region_names_list = []

  final_restore = all_restore if any(all_restore) else None

  if args.contextualize:
    if not region_names_list:
      region_names_list = _load_region_names(args.language)
    final_ctx = [
        _enrich_contextualization(c, region_names_list) for c in all_ctx
    ]
  else:
    final_ctx = [None] * len(all_ctx)

  embedding = None
  if args.embedding:
    embedding = generate_embedding(
        text=text, forward=forward, params=params, alphabet=alphabet
    )

  output = {
      'attribution': final_attr,
      'restoration': final_restore,
      'contextualize': final_ctx if args.contextualize else None,
      'embedding': embedding,
      'num_windows': len(windows),
  }

  _handle_output(
      output=output,
      args=args,
      final_attr=final_attr,
      final_restore=final_restore,
      final_ctx=final_ctx,
      embedding=embedding,
  )


def _parse_args() -> argparse.Namespace:
  """Parses and returns command-line arguments for the inference script."""
  parser = argparse.ArgumentParser(
      description='Run Predicting The Past inference.'
  )
  parser.add_argument(
      '--language',
      required=True,
      choices=['latin', 'greek'],
      help='Language of the input text.',
  )
  input_group = parser.add_mutually_exclusive_group(required=True)
  input_group.add_argument('--input', help='Input text string.')
  input_group.add_argument('--input_file', help='Path to input text file.')
  parser.add_argument(
      '--models_dir',
      default=_MODELS_DIR,
      help='Directory containing model files.',
  )
  parser.add_argument(
      '--output_json', default='', help='Save combined JSON to file.'
  )
  parser.add_argument('--restore_beam_width', type=int, default=100)
  parser.add_argument('--restore_max_len', type=int, default=15)
  parser.add_argument('--restore_temperature', type=float, default=1.0)
  parser.add_argument(
      '--window_overlap',
      type=float,
      default=0.33,
      help='Overlap fraction for long-text windowing.',
  )
  parser.add_argument(
      '--attribute',
      action='store_true',
      default=False,
      help='Run geographical and chronological attribution.',
  )
  parser.add_argument(
      '--contextualize',
      action='store_true',
      default=False,
      help='Run contextualization (parallel inscription retrieval).',
  )
  parser.add_argument(
      '--restore',
      action='store_true',
      default=False,
      help='Run text restoration (requires ? or # in input).',
  )
  parser.add_argument(
      '--contextualize_top_k',
      type=int,
      default=10,
      help='Number of top parallel inscriptions to return (default: 10).',
  )
  parser.add_argument(
      '--contextualize_exclude_test_valid',
      action='store_true',
      default=False,
      help=(
          'Exclude test/validation texts from contextualization '
          '(IDs ending in 3 or 4).'
      ),
  )

  parser.add_argument(
      '--embedding',
      action='store_true',
      default=False,
      help='Generate a text embedding vector.',
  )
  return parser.parse_args()


if __name__ == '__main__':
  main()
