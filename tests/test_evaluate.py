import numpy as np

from hotel_risk.evaluate import confusion_report


def test_confusion_counts_and_rates():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.6, 0.1, 0.1, 0.3, 0.4])
    r = confusion_report(y, p, threshold=0.5)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (2, 1, 1, 4)
    assert r["precision"] == round(2 / 3, 4)
    assert r["recall"] == round(2 / 3, 4)
    assert r["specificity"] == 0.8
    assert r["accuracy"] == round(6 / 8, 4)


def test_no_flags_does_not_divide_by_zero():
    r = confusion_report(np.array([0, 1, 0]), np.array([0.1, 0.2, 0.3]))
    assert r["tp"] == 0 and r["fp"] == 0
    assert r["precision"] == 0.0 and r["f1"] == 0.0
