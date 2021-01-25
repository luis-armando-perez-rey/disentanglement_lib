# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example script how to get started with research using disentanglement_lib.

To run the example, please change the working directory to the containing folder
and run:
>> python example.py

In this example, we show how to use disentanglement_lib to:
1. Train a standard VAE (already implemented in disentanglement_lib).
2. Train a custom VAE model_name.
3. Extract the mean representations for both of these models.
4. Compute the Mutual Information Gap (already implemented) for both models.
5. Compute a custom disentanglement metric for both models.
6. Aggregate the results.
7. Print out the final Pandas data frame with the results.
"""

# We group all the imports at the top.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from disentanglement_lib.methods.unsupervised import train

os.environ["CUDA_VISIBLE_DEVICES"]="2"
# 0. Settings
# ------------------------------------------------------------------------------
# By default, we save all the results in subdirectories of the following path.
base_path = "/home/luis/disentanglement_lib/results_models_data"

# By default, we do not overwrite output directories. Set this to True, if you
# want to overwrite (in particular, if you rerun this script several times).
overwrite = True

# 1. Train a standard VAE (already implemented in disentanglement_lib).
# ------------------------------------------------------------------------------

# We save the results in a `vae` subfolder.
dataset = "modelnet"
repetitions = 1
model_name_list = ["tcvae", "betavae", "factorvae"]

print("Running for {} repetitions".format(repetitions))
for model_name in model_name_list:
    print("Running {}".format(model_name))
    for repetition in range(repetitions):

        print("Training repetition {}".format(repetition))
        path_vae = os.path.join(base_path, dataset +"_" + model_name + "_" + str(repetition))
        model_path = os.path.join(path_vae, model_name)
        print("Model path {}".format(model_path))
        print("Dataset {}".format(dataset))
        print("Model name {}".format(model_name))
        print([dataset+"_" + model_name + ".gin"])
        train.train_with_gin(os.path.join(model_path, "model_name"), overwrite, ["model.gin"])
        train.train_with_gin(os.path.join(model_path, "model_name"), overwrite, [dataset+"_" + model_name + ".gin"])




