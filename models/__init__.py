import comet_ml
import gin
import trax
from models import models
from models.utils import get_trax_generator, train_model, predict, make_coalescent_heatmap

ReformerModel = trax.models.model_configure(models.ReformerModel)
GruModel = trax.models.model_configure(models.GruModel)
OfflineExperiment = gin.external_configurable(comet_ml.OfflineExperiment)
