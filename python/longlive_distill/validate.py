from __future__ import annotations
import argparse
from pathlib import Path
import onnx
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args();path=Path(a.model)
    if not path.is_file():raise SystemExit(f"missing exported model: {path}")
    m=onnx.load(str(path));onnx.checker.check_model(m)
    ins=[i.name for i in m.graph.input];outs=[o.name for o in m.graph.output]
    if ins!=['noisy_latents','timestep','prompt_embeds']:raise SystemExit(f"unexpected inputs: {ins}")
    if outs!=['noise_pred']:raise SystemExit(f"unexpected outputs: {outs}")
    print(f"{path}: valid ONNX, {path.stat().st_size} bytes, inputs={ins} outputs={outs}")
if __name__=='__main__':main()
