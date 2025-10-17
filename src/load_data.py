import argparse
import pathlib
import os
import shutil
import json


def _run(args):
	# Creating the output path
	os.makedirs(args.output_path)

	# Loading the data from "go-emotions-dataset" repository
	# to the format expected by the "distil_bert" repository

	# Copying id2label mapping
	shutil.copy(src=os.path.join(args.input_path, "id2label.json"),
				dst=args.output_path)

	# If there's no limit defined to the number of samples, we 
	# just copy the data
	if args.samples_limit == 0:
		# Train subset
		shutil.copy(src=os.path.join(args.input_path, "train_initial_model.json"),
					dst=os.path.join(args.output_path, "train.json"))
		# Validation subset
		shutil.copy(src=os.path.join(args.input_path, "validation.json"),
					dst=args.output_path)
		# Test subset
		shutil.copy(src=os.path.join(args.input_path, "test.json"),
					dst=args.output_path)
		
	# If there's a limit defined, we need to load each file individually
	else:
		# Train subset
		with open(os.path.join(args.input_path, "train_initial_model.json")) as f:
			train_data = json.loads(f.read())
		with open(os.path.join(args.output_path, "train.json"), mode='w') as f:
			f.write(json.dumps(train_data[:args.samples_limit]))
		# Validation subset
		with open(os.path.join(args.input_path, "validation.json")) as f:
			validation_data = json.loads(f.read())
		with open(os.path.join(args.output_path, "validation.json"), mode='w') as f:
			f.write(json.dumps(validation_data[:args.samples_limit]))
		# Test subset
		with open(os.path.join(args.input_path, "test.json")) as f:
			validation_data = json.loads(f.read())
		with open(os.path.join(args.output_path, "test.json"), mode='w') as f:
			f.write(json.dumps(validation_data[:args.samples_limit]))
	


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--input-path",
						type=pathlib.Path,
						required=True)

	parser.add_argument("--output-path", 
						type=pathlib.Path,
						required=True)

	parser.add_argument("--samples-limit",
						default=0,
						type=int,
						help="Maximum number of samples to include in train, "
							 "validation and test sets. Useful to prepare a small "
							 "dataset for development. If 0 (default), loads the "
							 "whole dataset.")

	args = parser.parse_args()

	_run(args)