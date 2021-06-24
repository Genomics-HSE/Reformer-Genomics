import comet_ml
import gin
from models import ReformerModel, get_trax_generator, train


if __name__ == '__main__':
    gin.parse_config_file("config.gin")
    
    train_gen = get_trax_generator()
    comet_exp = comet_ml.OfflineExperiment(project_name="population-genomics-new",
                                           workspace="kenenbek")
    
    train(model=ReformerModel(), train_gen=train_gen, comet_exp=comet_exp)
