import comet_ml
import gin
import argparse
from models import train_model, predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Genomics')
    parser.add_argument("--config", type=str, default="")
    action_parsers = parser.add_subparsers(title='actions', dest='action')
    train_parser = action_parsers.add_parser('train')
    
    predict_parser = action_parsers.add_parser('predict')
    predict_parser.add_argument('--path', type=str)
    args = parser.parse_args()

    gin.parse_config_file(args.config)
    
    if args.action == "train":
        train_model(model=gin.REQUIRED, train_gen=gin.REQUIRED, comet_exp=gin.REQUIRED,
                    lr=gin.REQUIRED, n_warmup_steps=gin.REQUIRED, n_steps_per_checkpoint=gin.REQUIRED,
                    output_dir=gin.REQUIRED, n_steps=gin.REQUIRED)
    elif args.action == "predict":
        predict(model_path=args.path, model=gin.REQUIRED,
                data_gen=gin.REQUIRED, num_genomes=gin.REQUIRED,
                plot_dir=gin.REQUIRED, plot_length=gin.REQUIRED
                )
    else:
        ValueError("Choose train or predict")
