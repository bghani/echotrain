"""Compatibility shim for legacy/models.py.

models.py was copied unchanged from the old repository and does
`from config import *`. These are the only names from the old config.py
that it uses. Do not add training settings here; those go in configs/*.yaml.
"""
import os  # noqa: F401  (models.py relies on `os` arriving via the star import)

SAMPLE_RATE = 32000
FMIN = 20
FMAX = 15000
