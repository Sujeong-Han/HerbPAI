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
    # 모든 "X in schema" 패턴을 안전하게 패치
    import re
    src = re.sub(
        r'if "(\w+)" in schema:',
        r'if isinstance(schema, dict) and "\1" in schema:',
        src
    )
    src = re.sub(
        r'elif "(\w+)" in schema:',
        r'elif isinstance(schema, dict) and "\1" in schema:',
        src
    )
    with open(utils_path, "w") as f:
        f.write(src)
    print(f"Patched: {utils_path}")
