import gradio_client.utils as u
import inspect, textwrap

src = inspect.getsource(u)
old = "        if \"const\" in schema:"
new = "        if isinstance(schema, dict) and \"const\" in schema:"
src = src.replace(old, new)

import os
path = u.__file__
with open(path, "w") as f:
    f.write(src)
print("Patched!")
