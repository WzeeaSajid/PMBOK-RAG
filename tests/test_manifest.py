"""Manifest shape validation before opening a persistent collection."""
import json
import unittest
from pathlib import Path

import _bootstrap

import config
import storage

from test_pipeline import TemporaryIndex


class ManifestTests(TemporaryIndex):
    def test_invalid_json_shapes_are_rejected(self):
        path = Path(config.INDEX_STATE_FILE)
        path.parent.mkdir(parents=True)
        for payload in (None, [], {}, {'collection': 4, 'signature': {}},
                        {'collection': '', 'signature': {}}, {'collection': 'index', 'signature': []}):
            with self.subTest(payload=payload):
                path.write_text(json.dumps(payload), encoding='utf-8')
                with self.assertRaises(storage.IndexErrorBase):
                    storage.read_state()


if __name__ == '__main__':
    unittest.main()
