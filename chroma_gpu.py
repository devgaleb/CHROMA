"""CHROMA GPU discovery and memory-aware worker admission."""
import json, subprocess

def discover():
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
    except (OSError,subprocess.SubprocessError):return []
    devices=[]
    for line in p.stdout.splitlines():
        parts=[x.strip() for x in line.split(',')]
        if len(parts)!=4:continue
        try:devices.append(dict(index=int(parts[0]),name=parts[1],total_mib=int(parts[2]),free_mib=int(parts[3])))
        except ValueError:continue
    return devices

def available(index,reserve_mib=1024):
    return next((max(0,d['free_mib']-reserve_mib) for d in discover() if d['index']==index),0)

if __name__=='__main__':print(json.dumps(discover(),indent=2))
