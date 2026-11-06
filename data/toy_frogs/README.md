# Toy frog dataset

A small dataset for trying out EchoTrain, used by
`notebooks/tutorial_toy_frogs.ipynb` and `configs/toy_frogs.yaml`.

- 5 species: *Bombina bombina*, *Bufo bufo*, *Epidalea calamita*,
  *Hyla arborea*, *Hyla meridionalis*
- 70 full field recordings (22 minutes): 8 per species for training, 3 for
  validation, 3 for test
- 5 s to 103 s long, at their original sample rates (32, 44.1 or 48 kHz),
  mono, Ogg Vorbis
- layout: `recordings/<split>/<species>/<recording>.ogg` (class folders)

The recordings are whole, not chunks. Cut them into 5 s chunks with
`tools/prepare_chunks.py` before training (the tutorial does this; the commands
are also at the top of `configs/toy_frogs.yaml`). The chunks go to
`data/toy_frogs_chunks/`, which is not tracked by git.

`recordings.csv` has one row per recording: the file path (relative to
`recordings/`), the species (`latin`), the split, length, sample rate, source,
recordist, licence, a link to the observation, date and place. It can also be
used as a manifest for `prepare_chunks.py --manifest`.

## Where the recordings come from

All recordings are from [iNaturalist](https://www.inaturalist.org), selected
from the AnuraSet Europe collection. Only recordings with a licence that allows
redistribution and reuse were used: CC BY (64), CC0 (5) and CC BY-SA (1). The
train / val / test split is the one of the full collection. Within each species
and split, recordings of 5 to 120 s at 32 kHz or more were picked at random.
They were mixed to mono and re-encoded as Ogg Vorbis to keep
the repository small.

## Licence and attribution

Each recording keeps the licence chosen by its recordist, listed in
`recordings.csv` together with the recordist's name and the observation link.
CC BY and CC BY-SA require attribution: when you reuse a recording, credit its
recordist and link to the observation. The CC BY-SA recording must be shared
under the same licence.
