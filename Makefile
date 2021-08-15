GPU = 1
CPU = 2
T = 600

config_file = big_config.gin

hse-run:
	echo "#!/bin/bash" > run.sh;
	echo "python main.py --config=$(config_file) train" >> run.sh;
	sbatch --gpus=$(GPU) -c $(CPU) -t $(T) run.sh;
	rm run.sh

print:
	echo $(config_file)
