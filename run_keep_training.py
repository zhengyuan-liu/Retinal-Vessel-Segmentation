###################################################
#
#   Script to launch the training
#
##################################################

import os, sys, time
import configparser

start = time.time()
config_name = None
if len(sys.argv) == 2:
    config_name = sys.argv[1]
else:
    print("Wrong Augment!")
    exit(1)

# config file to read from
config = configparser.RawConfigParser()
config.readfp(open(r'./' + config_name))
# ===========================================
# name of the experiment
name_experiment = config.get('experiment name', 'name')
nohup = config.getboolean('training settings', 'nohup')   # std output on log file?

run_GPU = '' if sys.platform == 'win32' else ' THEANO_FLAGS=device=gpu,floatX=float32 '

# create a folder for the results
result_dir = name_experiment
print("\n1. Create directory for the results (if not already existing)")
if os.path.exists(result_dir):
    print("Dir already existing")
elif sys.platform == 'win32':
    os.system('mkdir ' + result_dir)
else:
    os.system('mkdir -p ' + result_dir)

print("copy the configuration file in the results folder")
if sys.platform == 'win32':
    os.system('copy ' + config_name + ' .\\' + name_experiment+'\\' + name_experiment + '_configuration.txt')
else:
    os.system('cp ' + config_name + ' ./' + name_experiment + '/'+name_experiment+'_configuration.txt')

# run the experiment
if nohup:
    print("\n2. Run the training on GPU with nohup")
    os.system(run_GPU + ' nohup python -u ./src/retina_unet_keep_training.py ' + config_name +
              ' > ' + './' + name_experiment + '/' + name_experiment + '_training.nohup')
else:
    print("\n2. Run the training on GPU (no nohup)")
    os.system(run_GPU + ' python ./src/retina_unet_keep_training.py ' + config_name)

# Prediction/testing is run with a different script

end = time.time()
print("Running time (in sed): ", end-start)