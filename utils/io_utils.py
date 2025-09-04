import glob, os

def latest_joblib(models_dir):
    files = sorted(glob.glob(os.path.join(models_dir,'*')), key=os.path.getmtime, reverse=True)
    return files[0] if files else None
