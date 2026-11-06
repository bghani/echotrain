# legacy/

`models.py` is an unchanged copy of the model definitions from the old
repository (`batecho/models.py`). It holds many architectures; only a few
are wired into training, through `echotrain/model_zoo.py`.

The other files here exist only so that `models.py` can be imported as-is:

| file              | why it is here                                        |
|-------------------|-------------------------------------------------------|
| `root_modules.py` | unchanged copy; `models.py` does `from root_modules import *` |
| `config.py`       | shim; `models.py` does `from config import *`         |
| `util.py`         | shim; `models.py` does `from util import min_max_norm` |

`models.py` also imports `perchv2_pytorch` at the top. If that package is not
installed, `model_zoo.py` puts a placeholder in its place, so that only the
ONNX-based Perch models fail, and they fail with a clear error.

Once it is decided which models to keep, move them into `echotrain/models.py`
and delete this folder.
