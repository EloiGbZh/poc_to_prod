import unittest
import tempfile

from predict.predict.run import TextPredictionModel
from train.train import run
from preprocessing.preprocessing import utils
from unittest.mock import MagicMock
from train.tests import test_model_train

class TestPredict(unittest.TestCase):
    def test_predict(self):
        dict_params = {'batch_size': 2,
                       'epochs': 5,
                       'dense_dim': 64,
                       'min_samples_per_label': 1,
                       'verbose': 1}
        utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=test_model_train.load_dataset_mock())

        # we create a temporary file to store artefacts
        with tempfile.TemporaryDirectory() as model_dir:
            # run a training
            accuracy, artefacts_path = run.train('fake_dataset_path',
                                                 dict_params,
                                                 r"C:\Users\Eloi\Documents\EPF S7\POC to PROD\poc-to-prod-capstone\train\data\artefacts\2023-01-04-17-24-12",
                                                 True)

        model = TextPredictionModel.from_artefacts(artefacts_path)
        prediction = model.predict(["Is it possible to execute the procedure of a function in the scope of the caller?"], 1)

        self.assertEqual(['php'], prediction)