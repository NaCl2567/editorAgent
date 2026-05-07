from pathlib import Path
WORKDIR = Path.cwd()

def read_file(file_path: str, default: str = "") -> str:
    path = Path(file_path)

    if not path.exists():
        return default

    return path.read_text(encoding="utf-8")

def write_file(file_path: str, content: str) -> None:
    path = Path(file_path).expanduser()

    if path.is_absolute():
        raise ValueError(f"不允许写入绝对路径: {file_path}")

    full_path = (WORKDIR / path).resolve()

    if not str(full_path).startswith(str(WORKDIR.resolve())):
        raise ValueError(f"不允许写入工作目录之外的路径: {file_path}")

    full_path.parent.mkdir(parents=True, exist_ok=True)

    full_path.write_text(content, encoding="utf-8")




# import subprocess
# import tempfile
# from pathlib import Path


# WORKDIR = Path.cwd()


# def safe_resolve_path(file_path: str) -> Path:
#     path = Path(file_path).expanduser()

#     if path.is_absolute():
#         raise ValueError(f"不允许使用绝对路径: {file_path}")

#     full_path = (WORKDIR / path).resolve()

#     if not str(full_path).startswith(str(WORKDIR.resolve())):
#         raise ValueError(f"不允许访问工作目录之外的路径: {file_path}")

#     return full_path


# def read_file(file_path: str, default: str = "") -> str:
#     full_path = safe_resolve_path(file_path)

#     if not full_path.exists():
#         return default

#     return full_path.read_text(encoding="utf-8")


# def write_file(file_path: str, content: str) -> None:
#     full_path = safe_resolve_path(file_path)
#     full_path.parent.mkdir(parents=True, exist_ok=True)
#     full_path.write_text(content, encoding="utf-8")


# def apply_patch(patch: str) -> str:
#     """
#     使用 git apply 应用 unified diff patch。
#     先 --check，确认能应用后再真正 apply。
#     """
#     patch = patch.strip()

#     if not patch:
#         raise ValueError("patch 为空")

#     if "--- " not in patch or "+++ " not in patch or "@@" not in patch:
#         raise ValueError("patch 不是合法的 unified diff 格式")

#     with tempfile.NamedTemporaryFile(
#         mode="w",
#         encoding="utf-8",
#         suffix=".patch",
#         delete=False,
#     ) as f:
#         f.write(patch)
#         patch_file = f.name

#     patch_path = Path(patch_file)

#     try:
#         check_result = subprocess.run(
#             ["git", "apply", "--check", str(patch_path)],
#             cwd=WORKDIR,
#             text=True,
#             capture_output=True,
#         )

#         if check_result.returncode != 0:
#             raise RuntimeError(
#                 "patch 检查失败：\n"
#                 f"STDOUT:\n{check_result.stdout}\n\n"
#                 f"STDERR:\n{check_result.stderr}"
#             )

#         apply_result = subprocess.run(
#             ["git", "apply", str(patch_path)],
#             cwd=WORKDIR,
#             text=True,
#             capture_output=True,
#         )

#         if apply_result.returncode != 0:
#             raise RuntimeError(
#                 "patch 应用失败：\n"
#                 f"STDOUT:\n{apply_result.stdout}\n\n"
#                 f"STDERR:\n{apply_result.stderr}"
#             )

#         return "patch applied successfully"

#     finally:
#         patch_path.unlink(missing_ok=True)