"""Shared configuration fixture for public consumer and ownership tests."""


def configuration(**changes):
    result = {
        "name": "global-consumers", "seed": 13,
        "training": {"epochs": 2, "batch_size": 4, "learning_rate": .05,
                     "optimizer": "sgd", "momentum": 0., "lr_gamma": .5},
        "runtime": {"device": "cpu", "precision": "fp32"},
        "loader": {"num_workers": 0, "drop_last": True},
        "logging": {"level": "WARNING", "file_level": "DEBUG"},
        "schedules": {"coefficient": {"type": "constant", "value": .3}},
    }
    result.update(changes)
    return result
