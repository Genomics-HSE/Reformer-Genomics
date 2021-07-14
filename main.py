import comet_ml
import gin
import argparse
from models import ReformerModel, get_trax_generator, train, predict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Genomics')
    parser.add_argument("--config", type=str, default="")
    action_parsers = parser.add_subparsers(title='actions', dest='action')
    train_parser = action_parsers.add_parser('train')
    
    predict_parser = action_parsers.add_parser('predict')
    predict_parser.add_argument('--path', type=str)
    args = parser.parse_args()

    gin.parse_config_file(args.config)

    train_gen = get_trax_generator()
    comet_exp = comet_ml.OfflineExperiment(project_name="population-genomics-new",
                                           workspace="kenenbek",
                                           offline_directory="output/")
    
    if args.action == "train":
        train(model=ReformerModel(), train_gen=train_gen, comet_exp=comet_exp)
    elif args.action == "predict":
        predict_gen = get_trax_generator(random_seed=34)
        predict(model=ReformerModel(mode='predict'),
                model_path=args.path,
                data_generator=predict_gen
                )
    else:
        ValueError("Choose train or predict")
