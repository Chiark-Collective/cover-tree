import numpy as np
from covertreex.metrics.residual.host_backend import _point_decoder_factory


def test_point_decoder_accepts_indices_and_coords():
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    decoder = _point_decoder_factory(points)

    # Integer ids (1D and 2D)
    ids = np.array([2, 0], dtype=np.int64)
    np.testing.assert_array_equal(decoder(ids), np.array([2, 0], dtype=np.int64))
    ids_col = ids.reshape(-1, 1)
    np.testing.assert_array_equal(decoder(ids_col), np.array([2, 0], dtype=np.int64))
    float_ids = ids_col.astype(np.float32)
    np.testing.assert_array_equal(decoder(float_ids), np.array([2, 0], dtype=np.int64))

    # Coordinates map back to indices
    coords = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    np.testing.assert_array_equal(decoder(coords), np.array([1, 0], dtype=np.int64))

    # Unknown coordinates should raise
    bad_coords = np.array([[5.0, 5.0]], dtype=np.float64)
    try:
        decoder(bad_coords)
    except KeyError:
        pass
    else:
        raise AssertionError("decoder accepted unknown coordinate payload")
