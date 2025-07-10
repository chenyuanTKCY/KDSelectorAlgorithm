########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: root
# @file : train_deep_model
#
########################################################################

import argparse
import os
import re
# import silence_tensorflow.auto
from datetime import datetime

import numpy as np
import pandas as pd

from InfoBatch.infobatch import InfoBatch

from utils.train_deep_model_utils import ModelExecutioner, json_file

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.timeseries_dataset import create_splits, TimeseriesDataset
from utils.config import *
from eval_deep_model import eval_deep_model
# from infobatch import InfoBatch
import faiss
import time


def initialize_hash_buckets(dataset, nbits=14):
	print("Initializing hash buckets...")


	feature_vectors = np.array([data[0] for data in dataset], dtype=np.float32).reshape(len(dataset), -1)
	# print(f"Feature Vectors Shape: {feature_vectors.shape}, dtype: {feature_vectors.dtype}")

	d = feature_vectors.shape[1]
	index_lsh = faiss.IndexLSH(d, nbits)
	index_lsh.train(feature_vectors)
	# print(f"LSH Index initialized with dimension: {d} and nbits: {nbits}")

	hash_codes = np.zeros((len(dataset), (nbits + 7) // 8), dtype=np.uint8)
	index_lsh.sa_encode(feature_vectors, hash_codes)
	# print(f"Encoded Hash Codes Generated - Shape: {hash_codes.shape}, dtype: {hash_codes.dtype}")


	hash_bucket_ids = [code.tobytes() for code in hash_codes]
	# print("Hash buckets initialization completed.")

	# print("Sample Hash Bucket IDs:", hash_bucket_ids[:5])

	return hash_bucket_ids



def train_deep_model(
	data_path,
	model_name,
	split_per,
	seed,
	read_from_file,
	batch_size,
	model_parameters_file,
	epochs,
	alpha=None,
	eval_model=False,
	lambda_CL=0.5,
	temperature=1
):

	# Set up
	window_size = int(re.search(r'\d+', str(args.path)).group())
	device = 'cuda'
	save_runs = 'results/runs/'
	save_weights = 'results/weights/'
	inf_time = True 		# compute inference time per timeseries

	# Load the splits
	train_set, val_set, test_set = create_splits(
		data_path,
		split_per=split_per,
		seed=seed,
		read_from_file=read_from_file,
	)
	# For testing
	# train_set, val_set, test_set = train_set[:50], val_set[:10], test_set[:10]

	# Load the data
	print('----------------------------------------------------------------')
	training_data = TimeseriesDataset(data_path, fnames=train_set)


	# print("Starting hash bucket generation for training data...")
	from time import perf_counter
	tic = perf_counter()
	training_data.hash_codes = initialize_hash_buckets(training_data, nbits=15)
	toc = perf_counter()
	prune_time_all = toc - tic
	print(f"------------------------------+++++++++++++++++++{prune_time_all}+++++++++++++++++++++++++++++==----------------------")
	# print("Hash bucket generation for training data completed.")

	train_data = InfoBatch(training_data, epochs, 0.8, 0.875, 15, 8, hash_codes=training_data.hash_codes)
	val_data = TimeseriesDataset(data_path, fnames=val_set)
	test_data = TimeseriesDataset(data_path, fnames=test_set)

	# Create the data loaders
	# training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

	# training_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_data.sampler)

	validation_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

	# Compute class weights to give them to the loss function
	class_weights = training_data.get_weights_subset(device)

	# Read models parameters
	model_parameters = json_file(model_parameters_file)

	# Create the model, load it on GPU and print it
	model = deep_models[model_name.lower()](**model_parameters).to(device)
	model_fullname = f"{model_parameters_file.split('/')[-1].replace('.json', '')}_{window_size}"

	# Create the executioner object
	model_execute = ModelExecutioner(
		model=model,
		output_dim=args.output_dim,
		model_name=model_fullname,
		batch_size=batch_size,
		device=device,
		criterion=nn.CrossEntropyLoss(weight=class_weights).to(device),
		runs_dir=save_runs,
		weights_dir=save_weights,
		learning_rate=0.00001,
		lambda_CL=args.lambda_CL,
		alpha=args.alpha,
		temperature=args.temperature
	)

	SAVE_MERGED = f"results/merged/{args.model}_{args.output_dim}_{args.alpha}_{args.lambda_CL}_{args.temperature}_InfoFebruary_128"
	os.makedirs(SAVE_MERGED, exist_ok=True)
	# Check device of torch
	model_execute.torch_devices_info()

	# Run training procedure
	model, results,model_path, train_timestamp, prune_time = model_execute.train(
		n_epochs=epochs,
		# training_loader=training_loader,
		train_data=train_data,
		validation_loader=validation_loader,
		verbose=True,
	)
	prune_time_all += prune_time

	results.update({
		'model':args.model,
		'output_dim':args.output_dim,
		'lambda_CL':args.lambda_CL,
		'temperature':args.temperature,
		'prune_time_all':prune_time_all
	})
	# Save training stats
	timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
	df = pd.DataFrame.from_dict(results, columns=["training_stats"], orient="index")
	df.to_csv(os.path.join(save_done_training, f"{train_timestamp}_{model_fullname}_{timestamp}.csv"))
	df.to_csv(os.path.join(SAVE_MERGED, f"{train_timestamp}_{model_fullname}_{timestamp}.csv"))

	# Evaluate on test set or val set
	# if eval_model:
	# 	eval_set = test_set if len(test_set) > 0 else val_set
	# 	eval_deep_model(
	# 		data_path=data_path,
	# 		fnames=eval_set,
	# 		model_name=model_name,
	# 		model=model,
	# 		path_save=path_save_results,
	# 	)
	print(model_path)

	with open(f'{SAVE_MERGED}/weight.txt', 'w', encoding='utf-8') as file:

		file.write(model_path)

	time.sleep(2)
	return model_path


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog='run_experiment',
		description='This function is made so that we can easily run configurable experiments'
	)

	parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', required=True)
	parser.add_argument('-s', '--split', type=float, help='split percentage for train and val sets', default=0.7)
	parser.add_argument('-se', '--seed', type=int, default=None, help='Seed for train/val split')
	parser.add_argument('-f', '--file', type=str, help='path to file that contains a specific split', default=None)
	parser.add_argument('-m', '--model', type=str, help='model to run', required=True)
	parser.add_argument('-pa', '--params', type=str, help="a json file with the model's parameters", required=True)
	parser.add_argument('-b', '--batch', type=int, help='batch size', default=64)
	parser.add_argument('-ep', '--epochs', type=int, help='number of epochs', default=10)
	parser.add_argument('-e', '--eval-true', action="store_true", help='whether to evaluate the model on test data after training')
	parser.add_argument('-od', '--output_dim', type=int, default=256, help='Output dimension for the MLP layers')
	parser.add_argument('-l', '--lambda_CL', type=float, default=1.0, help='Weight for the contrastive loss component.')
	parser.add_argument('-t', '--temperature', type=float, default=1.0, help='Temperature parameter for softmax scaling.')
	parser.add_argument('-al','--alpha', type=float, default=0.5, help='alpha parameter for the loss function')

	args = parser.parse_args()
	train_deep_model(
		data_path=args.path,
		split_per=args.split,
		seed=args.seed,
		read_from_file=args.file,
		model_name=args.model,
		model_parameters_file=args.params,
		batch_size=args.batch,
		epochs=args.epochs,
		eval_model=args.eval_true,
		alpha=args.alpha,
		lambda_CL=args.lambda_CL,
		temperature=args.temperature
	)

