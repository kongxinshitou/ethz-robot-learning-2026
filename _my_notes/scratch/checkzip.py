import zipfile, hashlib, sys
z = zipfile.ZipFile(r"export\gpu_data_2026-11-16.zip")
names = z.namelist()
print("entries:", len(names))
tools = [n for n in names if n.startswith("_gpu_tools/")]
for n in tools: print("  ", n, z.getinfo(n).file_size)
root = r"GPU迁移_操作手册.md"
h1 = hashlib.sha256(z.read("_gpu_tools/GPU迁移_操作手册.md")).hexdigest()
h2 = hashlib.sha256(open("GPU迁移_操作手册.md","rb").read()).hexdigest()
print("手册一致:", h1 == h2)
s1 = hashlib.sha256(z.read("_gpu_tools/setup_gpu_env.sh")).hexdigest()
s2 = hashlib.sha256(open("setup_gpu_env.sh","rb").read()).hexdigest()
print("脚本一致:", s1 == s2)
print("MANIFEST:\n", z.read("MANIFEST.txt").decode("utf-8"))
