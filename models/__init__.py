import trax
from models import reformer

ReformerModel = trax.models.model_configure(reformer.ReformerModel)
