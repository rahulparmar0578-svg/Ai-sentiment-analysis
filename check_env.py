import sys, os, pathlib, subprocess, platform
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print(f"Platform: {platform.platform()}")
# Check available Python launchers
for p in [r"C:\Python", r"C:\Users\loq\AppData\Local\Programs\Python"]:
    if os.path.exists(p):
        print(f"Found: {p} -> {os.listdir(p)}")
    else:
        print(f"Not found: {p}")
# List C: root
try:
    print("C:/ listing (filtered):", [x for x in os.listdir("C:/") if "python" in x.lower() or "Python" in x][:20])
except Exception as e:
    print(e)
# Check pip
try:
    import pip
    print(f"pip: {pip.__version__}")
except Exception as e:
    print(f"pip import fail: {e}")
# Check existing packages
try:
    import importlib.metadata
    pkgs = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    for k in ["tensorflow","streamlit","scikit-learn","matplotlib","numpy"]:
        print(f"{k}: {pkgs.get(k, 'NOT INSTALLED')}  (also try case variants)")
    # print all
    print("All packages:", sorted(pkgs.keys())[:50])
except Exception as e:
    print(e)
# Check tensorflow compatibility with py314
print(f"Python version_info: {sys.version_info}")
# Check model/results
for d in ["sentiment-analysis/model","sentiment-analysis/results","model","results"]:
    if os.path.exists(d):
        print(f"{d}: {os.listdir(d)}")
