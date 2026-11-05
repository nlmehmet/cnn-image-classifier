"""Veri ayrımı, model seçimi ve tahmin için işlevsel regresyon testleri."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

import data
import evaluate
import train
from checkpoint import load_model
from model import SmallCNN


class TinyCIFAR:
    def __init__(self, root, train=True, download=True, transform=None):
        self.transform = transform

    def __len__(self):
        return 100

    def __getitem__(self, index):
        image = Image.new('RGB', (32, 32), (index, 50, 100))
        return self.transform(image), index % 10


class ValidationTests(unittest.TestCase):
    def test_split_is_reproducible_disjoint_and_validation_is_not_augmented(self):
        with patch.object(data.datasets, 'CIFAR10', TinyCIFAR):
            first, validation = data.get_training_loaders(augment=True, num_workers=0, seed=42)
            second, repeat = data.get_training_loaders(augment=True, num_workers=0, seed=42)
            _, different = data.get_training_loaders(num_workers=0, seed=43)
            self.assertEqual(len(first.dataset), 90)
            self.assertEqual(len(validation.dataset), 10)
            self.assertFalse(set(first.dataset.indices) & set(validation.dataset.indices))
            self.assertEqual(set(first.dataset.indices) | set(validation.dataset.indices), set(range(100)))
            self.assertEqual(validation.dataset.indices, repeat.dataset.indices)
            self.assertNotEqual(validation.dataset.indices, different.dataset.indices)
            self.assertEqual(first.dataset.indices, second.dataset.indices)
            self.assertTrue(any(isinstance(step, transforms.RandomCrop)
                                for step in first.dataset.dataset.transform.transforms))
            self.assertFalse(any(isinstance(step, (transforms.RandomCrop, transforms.RandomHorizontalFlip))
                                 for step in validation.dataset.dataset.transform.transforms))
            self.assertTrue(torch.equal(validation.dataset[0][0], validation.dataset[0][0]))

    def test_best_validation_checkpoint_and_test_used_only_after_training(self):
        torch.set_num_threads(2)
        dataset = TensorDataset(torch.randn(10, 3, 32, 32), torch.arange(10))
        training = DataLoader(dataset, batch_size=10)
        validation = DataLoader(dataset, batch_size=10)
        testing = DataLoader(dataset, batch_size=10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshots = []
            def measure(model, loader, device):
                if loader is validation:
                    snapshots.append({key: value.clone() for key, value in model.state_dict().items()})
                    return [0.6, 0.2][len(snapshots) - 1]
                self.assertIs(loader, testing)
                self.assertEqual(len(snapshots), 2)
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, snapshots[0][key]))
                return 0.3
            def open_test(**kwargs):
                self.assertEqual(len(snapshots), 2)
                return testing
            with patch.object(train, 'get_training_loaders', return_value=(training, validation)), \
                    patch.object(train, 'get_test_loader', side_effect=open_test) as get_test, \
                    patch.object(train, 'evaluate', side_effect=measure) as metric:
                train.main(['--name', 'validation', '--epochs', '2', '--workers', '0',
                            '--models-dir', str(root / 'models'), '--results-dir', str(root / 'results')])
            self.assertEqual(metric.call_count, 3)
            get_test.assert_called_once()
            result = json.loads((root / 'results/validation.json').read_text())
            self.assertEqual(result['best_val_acc'], 0.6)
            self.assertEqual(result['test_acc'], 0.3)
            self.assertEqual(result['history']['val_acc'], [0.6, 0.2])
            self.assertNotIn('test_acc', result['history'])
            checkpoint = torch.load(root / 'models/validation.pt', weights_only=True)
            self.assertEqual(checkpoint['epoch'], 1)

    def test_mixed_metrics_require_explicit_run_for_full_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = {'args': {'augment': False, 'bn': False, 'dropout': 0.0}}
            (root / 'old.json').write_text(json.dumps({**common, 'name': 'old', 'best_acc': .9,
                'history': {'train_loss': [1.], 'test_acc': [.9]}}))
            (root / 'new.json').write_text(json.dumps({**common, 'name': 'new', 'best_val_acc': .5,
                'test_acc': .4, 'history': {'train_loss': [1.], 'val_acc': [.5]}}))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                evaluate.main(['--results-dir', directory])
            self.assertEqual(error.exception.code, 2)
            evaluate.main(['--results-dir', directory, '--curves-only'])
            self.assertTrue((root / 'curves.png').is_file())


class PredictTests(unittest.TestCase):
    def test_cli_predicts_for_new_and_legacy_checkpoints(self):
        torch.set_num_threads(2)
        script = Path(__file__).resolve().parents[1] / 'src/predict.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'grayscale.png'
            Image.new('L', (70, 40), 100).save(image)
            for bn in (False, True):
                model = SmallCNN(use_bn=bn, dropout=.3)
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.zero_()
                    model.classifier[-1].bias[9] = 10
                model.eval()
                state = model.state_dict()
                for legacy in (False, True):
                    path = root / 'model.pt'
                    torch.save(state if legacy else {'state_dict': state,
                        'model_config': {'use_bn': bn, 'dropout': .3}}, path)
                    loaded = load_model(path)
                    with torch.inference_mode():
                        self.assertTrue(torch.equal(model(torch.zeros(1, 3, 32, 32)),
                                                    loaded(torch.zeros(1, 3, 32, 32))))
                    result = subprocess.run([sys.executable, str(script), '--image', str(image),
                                             '--model', str(path), '--top-k', '2'],
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(len(result.stdout.strip().splitlines()), 3)
                    self.assertIn('truck | %99.', result.stdout.splitlines()[1])

    def test_predict_reports_missing_file_and_invalid_top_k(self):
        script = Path(__file__).resolve().parents[1] / 'src/predict.py'
        for extra, message in [([], 'Dosya bulunamadı'), (['--top-k', '11'], '--top-k')]:
            result = subprocess.run([sys.executable, str(script), '--image', '/nonexistent/image.png',
                                     '--model', '/nonexistent/model.pt', *extra], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn(message, result.stderr)


if __name__ == '__main__':
    unittest.main()
