"""File-name conventions.

* The static trial is the c3d whose name contains ``Settings.static_token`` (case-insensitive).
* Every dynamic trial ends with one token per analysed leg, ``L<n>`` or ``R<n>`` (upper case),
  separated by ``_``, ``-`` or a space: ``cut02_L3.c3d`` analyses the left leg on plate 3,
  ``drop_jump_L1_R2.c3d`` both legs on plates 1 and 2. Only tokens at the END of the name count,
  so subject codes such as ``R01_cut01_L3`` are not mistaken for leg tokens.
  Plate numbers are 1-based, in the order of the force platforms stored in the c3d.
"""
from __future__ import annotations

import re
from pathlib import Path

_SEP = re.compile(r"[_\-\s]+")
_LEG = re.compile(r"^([LR])(\d{1,2})$")


def is_static(path: str | Path, static_token: str) -> bool:
    """True if the file name contains the static token (case-insensitive)."""
    return static_token.lower() in Path(path).stem.lower()


def parse_leg_plates(path: str | Path) -> dict[str, int]:
    """Analysed legs and their force plates from the trailing name tokens, e.g. ``{"left": 3}``.

    Returns {} if the name ends without a leg token. Raises ``ValueError`` if a leg is named
    twice or both legs are assigned to the same plate."""
    tokens = [t for t in _SEP.split(Path(path).stem) if t]
    out: dict[str, int] = {}
    for tok in reversed(tokens[1:]):            # the first token is never a leg token
        m = _LEG.match(tok)
        if not m:
            break
        leg, plate = ("left" if m.group(1) == "L" else "right"), int(m.group(2))
        if plate < 1:
            raise ValueError(f"{Path(path).name}: plate numbers start at 1")
        if leg in out:
            raise ValueError(f"{Path(path).name}: {leg} leg named twice")
        out[leg] = plate
    if len(out) == 2 and out["left"] == out["right"]:
        raise ValueError(f"{Path(path).name}: both legs assigned to plate {out['left']}")
    return dict(sorted(out.items()))
