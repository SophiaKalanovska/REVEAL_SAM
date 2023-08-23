# from innvestigate import analyzer
from . import analyzer
from .analyzer import create_analyzer
from .analyzer import NotAnalyzeableModelException
from .clusterRelevance import masks_from_heatmap
from .clusterRelevance import illustrate_clusters

# Disable pyflaks warnings:
assert analyzer
assert create_analyzer
assert NotAnalyzeableModelException
from innvestigate.backend.graph import model_wo_softmax