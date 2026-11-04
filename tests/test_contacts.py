import numpy as np

from markersub.contacts import select_contact

RATE = 1000.0


def _plate(segments, n=2000):
    fz = np.zeros(n)
    for a, b, f in segments:
        fz[a:b + 1] = f
    return fz


def test_single_contact():
    c = select_contact(_plate([(500, 800, 900.0)]), RATE, 20.0, 0.1)
    assert (c.td_s, c.to_s, c.rule) == (0.5, 0.8, "single")


def test_short_tap_is_ignored():
    c = select_contact(_plate([(300, 340, 300.0), (500, 800, 900.0)]), RATE, 20.0, 0.1)
    assert c.td_s == 0.5 and c.n_contacts == 1


def test_preloaded_plate_gives_landing():
    c = select_contact(_plate([(0, 700, 700.0), (1100, 1400, 1500.0)]), RATE, 20.0, 0.1)
    assert (c.td_s, c.rule) == (1.1, "preloaded_landing")


def test_loaded_at_start_is_flagged():
    c = select_contact(_plate([(0, 700, 700.0)]), RATE, 20.0, 0.1)
    assert c.rule == "loaded_at_start"


def test_two_contacts_largest_impulse():
    c = select_contact(_plate([(300, 420, 400.0), (700, 950, 900.0)]), RATE, 20.0, 0.1)
    assert (c.td_s, c.rule) == (0.7, "max_impulse")


def test_two_contacts_foot_on_plate():
    fz = _plate([(300, 420, 400.0), (700, 950, 900.0)])
    c = select_contact(fz, RATE, 20.0, 0.1, foot_on_plate=lambda pk: pk < 500)
    assert (c.td_s, c.rule) == (0.3, "foot_on_plate")
