import os, sys

utils_path = None
for path in sys.path:
    candidate = os.path.join(path, "gradio_client", "utils.py")
    if os.path.exists(candidate):
        utils_path = candidate
        break

if utils_path:
    with open(utils_path, "r") as f:
        src = f.read()
    src = src.replace(
        'if "const" in schema:',
        'if isinstance(schema, dict) and "const" in schema:'
    )
    src = src.replace(
        'if "enum" in schema:',
        'if isinstance(schema, dict) and "enum" in schema:'
    )
    with open(utils_path, "w") as f:
        f.write(src)
    print(f"Patched: {utils_path}")
