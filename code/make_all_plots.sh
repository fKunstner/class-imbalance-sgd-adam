echo "Before running, (see README.md)"
echo "  - Download the experiment data in the workspace directory"
echo "  - Specify the workspace directory in the environment variable"
echo "  - Install the required python packages and the library"

##
# Downloading from WandB
#
# For release code, download the experiment data from the github repository instead.
#
# python -m optexp.experiments --download
# python src/optexp/experiments/big_experiments.py --download
# python src/optexp/experiments/paper_figures.py

##
# Plotting
#
python -m optexp.experiments --plot
python -m optexp.experiments --plot-perclass
python src/optexp/experiments/big_experiments.py --plot --use_step
python src/optexp/experiments/big_experiments.py --plot-perclass --use_step

cd scripts
python plot_legends.py
python plot_modified_mnist.py
python plot_zipf_tokens.py
python plot_frequency_statistics.py

cd grad_vs_hessian
python main.py
python probs_over_time.py
python subset_hessian.py