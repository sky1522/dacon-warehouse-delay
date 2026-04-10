"""Drive <-> local 양방향 체크포인트 동기화."""
import os
import shutil
import glob

LOCAL_DIR = 'output'
DRIVE_DIR = '/content/drive/MyDrive/dacon_ckpt'

def sync_to_drive():
    """Local -> Drive 동기화."""
    os.makedirs(DRIVE_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(LOCAL_DIR, 'ckpt_*.pkl')):
        dst = os.path.join(DRIVE_DIR, os.path.basename(f))
        if not os.path.exists(dst) or os.path.getmtime(f) > os.path.getmtime(dst):
            shutil.copy(f, dst)
            print(f"  -> Drive: {os.path.basename(f)}")

def sync_from_drive():
    """Drive -> Local 동기화."""
    os.makedirs(LOCAL_DIR, exist_ok=True)
    if not os.path.exists(DRIVE_DIR):
        print("Drive not mounted")
        return
    for f in glob.glob(os.path.join(DRIVE_DIR, 'ckpt_*.pkl')):
        dst = os.path.join(LOCAL_DIR, os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy(f, dst)
            print(f"  <- Drive: {os.path.basename(f)}")

def status():
    """현재 상태 출력."""
    local_files = set(os.path.basename(f) for f in glob.glob(os.path.join(LOCAL_DIR, 'ckpt_*.pkl')))
    drive_files = set()
    if os.path.exists(DRIVE_DIR):
        drive_files = set(os.path.basename(f) for f in glob.glob(os.path.join(DRIVE_DIR, 'ckpt_*.pkl')))

    print(f"Local: {len(local_files)} files")
    print(f"Drive: {len(drive_files)} files")
    only_local = local_files - drive_files
    only_drive = drive_files - local_files
    if only_local:
        print(f"  Local only: {sorted(only_local)}")
    if only_drive:
        print(f"  Drive only: {sorted(only_drive)}")

if __name__ == '__main__':
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
    if cmd == 'push':
        sync_to_drive()
    elif cmd == 'pull':
        sync_from_drive()
    else:
        status()
