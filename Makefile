GPU = 1
CPU = 2
T = 600

hse-run:
	echo "#!/bin/bash" > run.sh;
	echo "python main.py" >> run.sh;
	sbatch --gpus=$(GPU) -c $(CPU) -t $(T) run.sh;
	rm run.sh