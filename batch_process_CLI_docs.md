# batch_process.py — CLI Documentation

A command-line utility for batch inspection and editing of WAV audio files (with limited FLAC renaming support), built on `librosa` and `soundfile`.

## Requirements

- Python 3
- `numpy`, `matplotlib`, `librosa`, `soundfile`

## Synopsis

```
python batch_process.py <path> <proc_type> [args...]
```

| Argument    | Description |
|-------------|-------------|
| `path`      | A single `.wav` file, **or** a directory to walk recursively. |
| `proc_type` | The operation to perform (see below). |
| `args...`   | Zero or more extra arguments, specific to `proc_type`. |

**Note:** All errors are caught silently and printed as a bare message — the script never raises a traceback or a non-zero-looking failure to the console beyond that printed line.

---

## How `path` is handled

- **If `path` is a file:** it must end in `.wav`, or nothing happens. Only single-file operations apply (`specshow`, `stats`, `slice`, `sil`, `csr`).
- **If `path` is a directory:** the script walks it recursively (`os.walk`). Two file types are considered:
  - `.wav` files — passed to whichever `proc_type` was selected.
  - `.flac` files — only relevant to the `names` operation.

---

## Operations (`proc_type`)

### `specshow`
Displays a 3-panel plot (waveform, linear-frequency spectrogram, mel spectrogram) for each `.wav` file found, using `matplotlib`. Blocks on `plt.show()` for each file — you must close the window to continue to the next file.

```
python batch_process.py song.wav specshow
python batch_process.py ./audio_folder specshow
```

No extra args.

---

### `stats`
Reports duration, channel count, sample rate, per-channel peak level (with timestamp), and per-channel RMS level.

- **Single file:** prints results directly to the console.
- **Directory:** prints results per file to the console **and** writes a summary report to a `.txt` file (same name as the folder, placed next to it) once processing finishes.

```
python batch_process.py track.wav stats
python batch_process.py ./audio_folder stats [sortby]
```

| Arg (dir mode) | Description |
|---|---|
| `sortby` (optional) | One of `name`, `len`, `peak`, `rms`. Controls sort order of the written `.txt` report. Any other value (or omission) leaves files in the order they were walked. |

**Note:** Peak/RMS computation loops sample-by-sample in pure Python, so this is slow on large files/folders.

---

### `slice`
Trims audio by removing a number of seconds from the start and/or end, then **overwrites the original file**.

```
python batch_process.py track.wav slice [first] [last]
python batch_process.py ./audio_folder slice [first] [last]
```

| Arg | Default | Description |
|---|---|---|
| `first` | `0` | Seconds to remove from the start. |
| `last`  | `0` | Seconds to remove from the end. |

Output is re-saved as 32-bit float WAV (`subtype="FLOAT"`), regardless of the original bit depth.

---

### `sil`
Adds silence padding to the start and/or end of a file, then **overwrites the original file**.

```
python batch_process.py track.wav sil [first] [last]
python batch_process.py ./audio_folder sil [first] [last]
```

| Arg | Default | Description |
|---|---|---|
| `first` | `0` | Seconds of silence to prepend. |
| `last`  | `0` | Seconds of silence to append. |

Output is re-saved as 32-bit float WAV.

---

### `csr`
Rewrites the file with a new sample-rate value, then **overwrites the original file**.

```
python batch_process.py track.wav csr <newsr>
python batch_process.py ./audio_folder csr <newsr>
```

| Arg | Default | Description |
|---|---|---|
| `newsr` | `0` | New sample rate to write into the file header. |

**⚠️ Important:** This does **not** resample the audio data — it loads the file at its original rate and re-saves it tagged with `newsr`. The result will play back at the wrong speed/pitch unless you separately resample the samples yourself. Effectively this only relabels the sample rate.

---

### `names` *(directory mode only, `.flac` files only)*
Renames every `.flac` file in the tree by appending its parent folder's name to the filename, and **renames the file on disk**.

```
python batch_process.py ./audio_folder names
```

Example: `takes/session1/vocal.flac` → `takes/session1/vocal-session1.flac`

No extra args. Has no effect on `.wav` files, and does nothing at all when `path` is a single file.

---

### `join` *(directory mode only)*
Concatenates every `.wav` file found (in `os.walk` order — not guaranteed alphabetical) into a single audio stream, and writes it as one new file next to the source folder, using the first file's name with a special-character prefix.

```
python batch_process.py ./audio_folder join
```

No extra args. All files should share the same sample rate and channel layout; the script does not resample or reshape mismatched files before concatenating.

---

### `match` *(directory mode only)*
Groups `.wav` files into consecutive batches of size `nmatch` (in walk order) and, within each group, zero-pads every file shorter than the group's longest file so all files in the group share the same length. **Overwrites the padded files in place.**

```
python batch_process.py ./audio_folder match <nmatch>
```

| Arg | Description |
|---|---|
| `nmatch` | Number of files per group. |

If the number of `.wav` files found isn't an exact multiple of `nmatch`, the trailing partial group is silently dropped (never processed).

---

### `mix` *(directory mode only)*
Groups `.wav` files into consecutive batches of size `nmatch` (in walk order), sums the waveforms in each group, and writes the mixed result using the `nth` file's name (within that group) with a special-character prefix.

```
python batch_process.py ./audio_folder mix <nmatch> <nth>
```

| Arg | Description |
|---|---|
| `nmatch` | Number of files per group. |
| `nth`    | Index (0-based) of the file in the group whose name/path is used for the output filename. |

**Requirements:** all files in a group should already be the same length and sample rate — `mix` does not pad or resample before summing (unlike `match`). Clipping is possible since it's a raw sum with no normalization.

---

## Quick reference

| `proc_type` | Scope | Extra args | Modifies files? |
|---|---|---|---|
| `specshow` | file or dir | — | No |
| `stats` | file or dir | `[sortby]` (dir only) | No (writes a `.txt` report in dir mode) |
| `slice` | file or dir | `[first] [last]` | Yes (overwrite) |
| `sil` | file or dir | `[first] [last]` | Yes (overwrite) |
| `csr` | file or dir | `[newsr]` | Yes (overwrite, relabel only) |
| `names` | dir only | — | Yes (renames `.flac` files) |
| `join` | dir only | — | No (writes new file) |
| `match` | dir only | `<nmatch>` | Yes (overwrite, for padded files) |
| `mix` | dir only | `<nmatch> <nth>` | No (writes new file) |

## Known quirks / gotchas

- Errors are swallowed by a bare `except Exception as err: print(err)` — check console output carefully; there's no exit code signal of failure.
- `slice`, `sil`, and `csr` **overwrite the source file** with no backup — copy your files first if you want to keep originals.
- `csr` changes only the metadata sample rate, not the actual audio content — it is not a resampler.
- `match` drops any incomplete trailing group instead of processing it.
- File processing order in directory mode follows `os.walk`, which is filesystem-dependent and not alphabetically guaranteed.
- The `join`/`mix` output filename prefix is a non-ASCII character (`chr(148206)`), which may not render or sort as expected in all file managers/terminals.
