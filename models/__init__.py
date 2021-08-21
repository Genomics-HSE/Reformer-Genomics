import comet_ml
import gin
import trax
from models import models
from models.utils import get_trax_generator, train_model, predict_model, make_coalescent_heatmap

ReformerModel = trax.models.model_configure(models.ReformerModel)
GruModel = trax.models.model_configure(models.GruModel)

Embedding = trax.layers.layer_configure(models.Embedding)
CNNEmbedding = trax.layers.layer_configure(models.CNNEmbedding)

OfflineExperiment = gin.external_configurable(comet_ml.OfflineExperiment)

train_model = gin.external_configurable(train_model)
predict_model = gin.external_configurable(predict_model)
get_trax_generator = gin.external_configurable(get_trax_generator)
