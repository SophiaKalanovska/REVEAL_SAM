import pytest
import tensorflow as tf

import innvestigate.analyzer

from tests import dryrun

# Dict that maps test name to tuple of method and kwargs
methods = {
    "DeepTaylor": (innvestigate.analyzer.DeepTaylor, {}),
    "BoundedDeepTaylor": (innvestigate.analyzer.BoundedDeepTaylor, {"low": -1, "high": 1}),
}


@pytest.mark.deeptaylor
@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("method, kwargs", methods.values(), ids=list(methods.keys()))
def test_fast(method, kwargs):
    tf.keras.backend.clear_session()

    def analyzer(model):
        return method(model, **kwargs)

    dryrun.test_analyzer(analyzer, "trivia.*:mnist.log_reg")


@pytest.mark.deeptaylor
@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("method, kwargs", methods.values(), ids=list(methods.keys()))
def test_fast_serialize(method, kwargs):
    tf.keras.backend.clear_session()

    def analyzer(model):
        return method(model, **kwargs)

    dryrun.test_serialize_analyzer(analyzer, "trivia.*:mnist.log_reg")


@pytest.mark.deeptaylor
@pytest.mark.mnist
@pytest.mark.precommit
@pytest.mark.parametrize("method, kwargs", methods.values(), ids=list(methods.keys()))
def test_precommit(method, kwargs):
    tf.keras.backend.clear_session()

    def analyzer(model):
        return method(model, **kwargs)

    dryrun.test_analyzer(analyzer, "mnist.*")


@pytest.mark.deeptaylor
@pytest.mark.slow
@pytest.mark.application
@pytest.mark.imagenet
@pytest.mark.parametrize("method, kwargs", methods.values(), ids=list(methods.keys()))
def test_imagenet(method, kwargs):
    tf.keras.backend.clear_session()

    def analyzer(model):
        return method(model, **kwargs)

    dryrun.test_analyzer(analyzer, "imagenet.*")
