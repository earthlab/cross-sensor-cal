import argparse
import sys
from pathlib import Path

from src.neon_to_envi import neon_to_envi
from src.topo_and_brdf_correction import topo_and_brdf_correction
from src.convolution_resample import resample as convolution_resample


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_nevi = sub.add_parser("neon_to_envi")
    p_nevi.add_argument("--images", nargs="+", required=True)
    p_nevi.add_argument("--output_dir", required=True)
    p_nevi.add_argument("--anc", action="store_true")

    p_topo = sub.add_parser("topo")
    p_topo.add_argument("--config", required=True)

    p_res = sub.add_parser("resample")
    p_res.add_argument("--dir", required=True)

    args = parser.parse_args()

    if args.cmd == "neon_to_envi":
        neon_to_envi(images=args.images, output_dir=args.output_dir, anc=args.anc)
    elif args.cmd == "topo":
        topo_and_brdf_correction(args.config)
    elif args.cmd == "resample":
        convolution_resample(Path(args.dir))
    else:  # pragma: no cover - argparse enforces choices
        parser.error("unknown command")
    return 0


if __name__ == "__main__":
    sys.exit(main())
