"""Exact nested state comparisons shared by training and recovery tests."""

import torch


def assert_state_equal(test, actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        test.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            assert_state_equal(test, actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        test.assertEqual(len(actual), len(expected))
        for value, reference in zip(actual, expected):
            assert_state_equal(test, value, reference)
    else:
        test.assertEqual(actual, expected)
