#!/usr/bin/env python3
"""Build a NeMo JSON manifest from wav.scp, run EncDecSpeakerLabelModel inference, write Kaldi vectors."""

from __future__ import annotations

import argparse
import json
import os
import sys


def wav_scp_to_manifest(
    wav_scp_path: str, manifest_path: str
) -> tuple[list[str], list[str]]:
    import soundfile as sf

    utts: list[str] = []
    paths: list[str] = []
    with open(wav_scp_path, encoding="utf-8") as fin, open(
        manifest_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            utt, apath = parts
            apath = os.path.abspath(apath)
            if not os.path.isfile(apath):
                raise FileNotFoundError(f"Missing wav for utt {utt}: {apath}")
            info = sf.info(apath)
            dur = float(info.duration)
            rec = {"audio_filepath": apath, "duration": dur, "label": "0"}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            utts.append(utt)
            paths.append(apath)
    return utts, paths


def extract_sequential(model, utts: list[str], wav_paths: list[str], device: str):
    import numpy as np
    import torch
    from tqdm import tqdm

    out: list = []
    model.eval()
    with torch.no_grad():
        for _, p in tqdm(
            list(zip(utts, wav_paths)),
            desc="get_embedding (sequential)",
            file=sys.stderr,
        ):
            emb = model.get_embedding(p)
            vec = emb.squeeze().detach().cpu().numpy().reshape(-1).astype("float32")
            out.append(vec)
    return np.stack(out, axis=0)


def main():
    import numpy as np
    import torch
    import kaldiio
    import nemo.collections.asr as nemo_asr

    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_scp", required=True, help="Kaldi wav.scp (utt wav_path)")
    ap.add_argument(
        "--output_dir",
        required=True,
        help="Directory for xvector.ark, xvector.scp, and manifest json",
    )
    ap.add_argument(
        "--model_name",
        default="nvidia/speakerverification_en_titanet_large",
        help="NeMo from_pretrained id (see Hugging Face model card)",
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument(
        "--device",
        default=None,
        help="cuda | cuda:0 | cpu (default: cuda if available else cpu)",
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "manifest.json")

    utts, wav_paths = wav_scp_to_manifest(args.wav_scp, manifest_path)
    if not utts:
        print("No utterances in wav.scp", file=sys.stderr)
        sys.exit(1)

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading TitaNet: {args.model_name} (device={device})", file=sys.stderr)
    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(args.model_name)
    model = model.to(device)
    model.eval()

    embs_np = None
    try:
        embs, _, _, _ = model.batch_inference(
            manifest_path, batch_size=args.batch_size, device=device
        )
        embs_np = np.asarray(embs)
        if embs_np.ndim == 3:
            embs_np = embs_np.reshape(embs_np.shape[0], -1)
        elif embs_np.ndim != 2:
            embs_np = embs_np.reshape(len(utts), -1)
    except Exception as e:
        print(
            f"batch_inference failed ({e}); falling back to per-file get_embedding.",
            file=sys.stderr,
        )
        embs_np = extract_sequential(model, utts, wav_paths, device)

    if embs_np.shape[0] != len(utts):
        raise RuntimeError(
            f"Embedding count {embs_np.shape[0]} != utterances {len(utts)}"
        )

    ark = os.path.join(args.output_dir, "xvector.ark")
    scp = os.path.join(args.output_dir, "xvector.scp")
    with kaldiio.WriteHelper(f"ark,scp:{ark},{scp}") as writer:
        for utt, row in zip(utts, embs_np):
            vec = np.asarray(row, dtype=np.float32).reshape(-1)
            writer(utt, vec)

    print(f"Wrote {len(utts)} embeddings to {scp}", file=sys.stderr)


if __name__ == "__main__":
    main()
