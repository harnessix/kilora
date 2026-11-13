# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
import tempfile
import types
from pathlib import Path

from safetensors.torch import save_file

TESTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(TESTS.parent / "src"))
from test_chunk_plugin import _record, setup


def main():
    if len(sys.argv) > 1:
        source = Path(sys.argv[1]).read_text()
    else:
        source = subprocess.run(
            [
                "git",
                "show",
                "editor/t607d:plugins/kilora-editor-chunk/src/kilora_editor_chunk/plugin.py",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    module = types.ModuleType("kilora_editor_chunk._t607d_reference")
    module.__package__ = "kilora_editor_chunk"
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, "t607d/plugin.py", "exec"), module.__dict__)
        with tempfile.TemporaryDirectory() as folder:
            fixture = setup.__wrapped__(Path(folder))
            try:
                session, path, target = next(fixture)
                baseline = module.ChunkLive(str(path))
                baseline.check(session)
                baseline.configure({"path_merge": False})
                result = baseline.next_state(session.state, _record(session, target))
                tensors = {
                    f"editor/{key}": value.clone().contiguous()
                    for key, value in session.editor_state.state_tensors().items()
                }
                tensors.update(
                    {
                        f"result/{hook}/{key}": value.clone().contiguous()
                        for hook, parts in result.items()
                        for key, value in parts.items()
                    }
                )
                save_file(tensors, str(TESTS / "data" / "t607d_merge_off_reference.safetensors"))
            finally:
                fixture.close()
    finally:
        sys.modules.pop(module.__name__, None)


if __name__ == "__main__":
    main()
