import trax
from models import reformer
from models.utils import get_trax_generator, train

ReformerModel = trax.models.model_configure(reformer.ReformerModel)
