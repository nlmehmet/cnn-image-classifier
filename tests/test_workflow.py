"""İnternet gerektirmeyen komut ve eğitim/değerlendirme kontrolleri."""
import contextlib
import io
import json
from pathlib import Path
import sys
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from torch.utils.data import DataLoader, TensorDataset
import train
import evaluate


class WorkflowTests(unittest.TestCase):
    def test_invalid_training_arguments(self):
        for options in [['--epochs', '0'], ['--workers', '-1'],
                        ['--dropout', '1'], ['--lr', 'nan'],
                        ['--name', '../outside']]:
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    train.parse_args(['--name', 'test', *options])
                self.assertEqual(error.exception.code, 2)

    def test_batch_script_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = root / "record.py"
            log = root / "commands.jsonl"
            recorder.write_text(
                "#!/usr/bin/env python3\nimport json, os, sys\n"
                "with open(os.environ['COMMAND_LOG'], 'a') as stream:\n"
                "    stream.write(json.dumps([os.getcwd(), sys.argv[1:]]) + '\\n')\n"
            )
            recorder.chmod(0o755)
            project = Path(__file__).resolve().parents[1]
            subprocess.run(['bash', str(project / 'run_all.sh')], cwd=root, check=True,
                           env={**os.environ, 'PYTHON': str(recorder), 'EPOCHS': '1',
                                'RUN_DIR': 'runs/test space', 'COMMAND_LOG': str(log)})
            commands = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(commands), 4)
            self.assertTrue(all(cwd == str(project) for cwd, _ in commands))
            for _, args in commands:
                self.assertEqual(args[args.index('--results-dir') + 1], 'runs/test space/results')
                self.assertEqual(args[args.index('--models-dir') + 1], 'runs/test space/models')
            self.assertEqual(commands[-1][1][0], 'src/evaluate.py')

    def test_training_and_evaluation_with_custom_paths(self):
        torch.set_num_threads(2)
        dataset = TensorDataset(torch.randn(20, 3, 32, 32), torch.arange(20) % 10)
        loader = DataLoader(dataset, batch_size=10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, models, plots = [root / name for name in ('results', 'models', 'plots')]
            with patch.object(train, 'get_loaders', return_value=(loader, loader)) as load:
                # Zero accuracy must still save the first checkpoint.
                with patch.object(train, 'evaluate', return_value=0.0):
                    train.main(['--name', 'test', '--epochs', '1', '--bn',
                                '--dropout', '0.3', '--workers', '0',
                                '--results-dir', str(results), '--models-dir', str(models),
                                '--data-dir', str(root / 'data')])
                self.assertEqual(load.call_args.kwargs['num_workers'], 0)
                self.assertEqual(load.call_args.kwargs['root'], str(root / 'data'))
            self.assertTrue((models / 'test.pt').is_file())
            result = json.loads((results / 'test.json').read_text())
            self.assertEqual(result['best_acc'], 0.0)
            self.assertEqual(len(result['history']['train_loss']), 1)
            with patch.object(evaluate, 'get_loaders', return_value=(loader, loader)):
                evaluate.main(['--results-dir', str(results), '--models-dir', str(models),
                               '--output-dir', str(plots), '--workers', '0'])
            for name in ('curves.png', 'confusion_matrix.png'):
                self.assertGreater((plots / name).stat().st_size, 1000)
            (models / 'test.pt').unlink()
            with patch.object(evaluate, 'get_loaders', side_effect=AssertionError('download')):
                evaluate.main(['--results-dir', str(results), '--curves-only'])
            with self.assertRaisesRegex(SystemExit, 'Model bulunamadı'):
                evaluate.main(['--results-dir', str(results), '--models-dir', str(models)])


if __name__ == '__main__':
    unittest.main()
