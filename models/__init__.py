import trax
from models import reformer
from models.utils import get_trax_generator, train, predict, make_coalescent_heatmap

ReformerModel = trax.models.model_configure(reformer.ReformerModel)
