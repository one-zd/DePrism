"""Comprehensive and reproducible evaluator for DePrism image outputs.

This evaluator supersedes the legacy ``evaluate_images.py`` and
``evaluate_batch.py`` workflows while keeping their general Excel/CSV batch
interface.  It incorporates the CLIP embedding cosine metrics and normalized
NIMA variants used by the TIP supplementary experiments.

Run this file from the repository root.  Use ``--run LABEL=PATH`` more than
once to evaluate several methods in one invocation, or use the single-run
``--generated-dir`` interface.
"""

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NIMA_VARIANTS = (
    "no_norm_center_crop",
    "center_crop_norm",
    "resize_direct_norm",
    "five_crop_norm",
)
NIMA_TARGETS = ("generated", "reference", "comparison")
GLOB_CHARS = "*?["


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DePrism outputs with paired image metrics, normalized "
            "CLIP-T/CLIP-I cosine similarity, NIMA, and dataset-level FID."
        )
    )
    parser.add_argument(
        "--metadata",
        "--excel_path",
        dest="metadata",
        required=True,
        help="Metadata .xlsx or .csv containing image_name and prompt columns.",
    )
    parser.add_argument(
        "--ref-dir",
        "--ref_dir",
        dest="ref_dir",
        required=True,
        help="Directory containing raw/reference images.",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--generated-dir",
        "--generated_dir",
        dest="generated_dir",
        help="One method output root; each sample is stored in <root>/<image_stem>/.",
    )
    source.add_argument(
        "--run",
        action="append",
        metavar="LABEL=PATH",
        help="Method label and output root. Repeat to evaluate multiple methods.",
    )
    parser.add_argument(
        "--method-name",
        default="DePrism",
        help="Method label used with --generated-dir (default: DePrism).",
    )
    parser.add_argument(
        "--generated-name",
        "--generated_name",
        dest="generated_name",
        default="final_fused_img.png",
        help="Generated filename or glob relative to each sample directory.",
    )
    parser.add_argument(
        "--comparison-name",
        default=None,
        help=(
            "Optional Standard/baseline filename or glob in each sample directory, "
            "for example controlnet_gt.png."
        ),
    )
    parser.add_argument(
        "--comparison-label",
        default="standard",
        help="Column label for --comparison-name (default: standard).",
    )
    parser.add_argument(
        "--glob-policy",
        choices=("error", "first"),
        default="error",
        help="How to handle a filename glob matching multiple images.",
    )

    parser.add_argument("--image-column", default="image_name")
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--style-column", default="style_prompt")
    parser.add_argument("--category-column", default="category")
    parser.add_argument("--case-column", default="case_number")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Optional category whitelist.",
    )
    parser.add_argument(
        "--case-numbers",
        nargs="+",
        default=None,
        help="Optional exact case-number whitelist.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--missing-policy",
        choices=("skip", "error"),
        default="skip",
        help="Skip missing samples with a report, or stop immediately.",
    )

    parser.add_argument(
        "--clip-model",
        default="openai/clip-vit-large-patch14",
        help="Local CLIP path or Hugging Face model identifier.",
    )
    parser.add_argument(
        "--nima-ckpt",
        default="NIMA/snapshots/epoch-82.pth",
        help="NIMA checkpoint path.",
    )
    parser.add_argument(
        "--nima-variants",
        nargs="+",
        choices=NIMA_VARIANTS,
        default=["five_crop_norm"],
        help="NIMA preprocessing variants; NIMA-5C is the default.",
    )
    parser.add_argument(
        "--nima-targets",
        nargs="+",
        choices=NIMA_TARGETS,
        default=list(NIMA_TARGETS),
        help="Images on which NIMA is evaluated.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a concrete Torch device such as cuda:0.",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Transformers to download CLIP when the model is not local.",
    )
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--skip-nima", action="store_true")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--skip-fid", action="store_true")

    parser.add_argument(
        "--output-dir",
        default="scripts/comprehensive_evaluation",
        help="Directory for CSV, Markdown, and JSON outputs.",
    )
    parser.add_argument("--output-prefix", default="comprehensive")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    """Return a repository-relative path without requiring Python 3.9."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def safe_label(value: str) -> str:
    label = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    label = "_".join(part for part in label.split("_") if part)
    if not label:
        raise ValueError("comparison label must contain at least one letter or number")
    return label


def parse_runs(args: argparse.Namespace) -> List[Tuple[str, Path]]:
    if args.run:
        runs = []
        labels = set()
        for spec in args.run:
            if "=" not in spec:
                raise ValueError("--run must use LABEL=PATH syntax: {}".format(spec))
            label, raw_path = spec.split("=", 1)
            label = label.strip()
            if not label or not raw_path.strip():
                raise ValueError("--run requires non-empty LABEL and PATH")
            if label in labels:
                raise ValueError("duplicate --run label: {}".format(label))
            labels.add(label)
            runs.append((label, project_path(raw_path.strip())))
        return runs
    return [(args.method_name, project_path(args.generated_dir))]


def read_metadata(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("metadata must be .xlsx, .xls, or .csv: {}".format(path))


def nonempty_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def ensure_image_name(name: str) -> Sequence[str]:
    path = Path(name)
    if path.suffix or any(ch in name for ch in GLOB_CHARS):
        return [name]
    return [name, name + ".png"]


def resolve_sample_image(
    sample_dir: Path, name: Optional[str], glob_policy: str
) -> Tuple[Optional[Path], Optional[str]]:
    if not name:
        return None, None

    for candidate in ensure_image_name(name):
        if any(ch in candidate for ch in GLOB_CHARS):
            matches = sorted(
                path for path in sample_dir.glob(candidate) if path.is_file()
            )
            if len(matches) > 1 and glob_policy == "error":
                return None, "pattern matched multiple files: {}".format(candidate)
            if matches:
                return matches[0], None
        else:
            path = sample_dir / candidate
            if path.is_file():
                return path, None
    return None, "file not found: {}".format(name)


def load_rgb(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if size:
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize((size, size), resampling)
    return image


def paired_pixel_metrics(
    image_a: Image.Image, image_b: Image.Image
) -> Dict[str, float]:
    array_a = np.asarray(image_a)
    array_b = np.asarray(image_b)
    return {
        "psnr": float(peak_signal_noise_ratio(array_a, array_b, data_range=255)),
        "ssim": float(
            structural_similarity(array_a, array_b, channel_axis=-1, data_range=255)
        ),
    }


def prefixed(prefix: str, metrics: Dict[str, object]) -> Dict[str, object]:
    return {"{}_{}".format(prefix, key): value for key, value in metrics.items()}


class MetricRuntime:
    """Lazily loads only the requested heavyweight metric models."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.torch = None
        self.transforms = None
        self.device = None
        self.clip_model = None
        self.clip_processor = None
        self.nima_model = None
        self.lpips_model = None
        self.fid_class = None
        self.fid_transform = None
        self.image_feature_cache: Dict[str, object] = {}
        self.text_feature_cache: Dict[str, object] = {}
        self.nima_cache: Dict[Tuple[str, str], Dict[str, object]] = {}
        self.fid_metrics: Dict[Tuple[str, str, str], object] = {}
        self.fid_errors: Dict[Tuple[str, str, str], str] = {}
        self.versions: Dict[str, str] = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }
        self._load_requested_models()

    def _load_requested_models(self) -> None:
        needs_torch = not (
            self.args.skip_clip
            and self.args.skip_nima
            and self.args.skip_lpips
            and self.args.skip_fid
        )
        if not needs_torch:
            return

        import torch
        from torchvision import transforms as transforms

        self.torch = torch
        self.transforms = transforms
        self.device = self._select_device(self.args.device)
        self.versions["torch"] = torch.__version__

        if not self.args.skip_clip:
            from transformers import CLIPModel, CLIPProcessor
            import transformers

            clip_path = project_path(self.args.clip_model)
            source = str(clip_path if clip_path.exists() else self.args.clip_model)
            local_only = not self.args.allow_model_download
            self.clip_model = CLIPModel.from_pretrained(
                source, local_files_only=local_only
            ).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(
                source, local_files_only=local_only
            )
            self.clip_model.eval()
            self.versions["transformers"] = transformers.__version__

        if not self.args.skip_nima:
            import torchvision
            import torchvision.models as tv_models
            from NIMA.model.model import NIMA

            try:
                base_model = tv_models.vgg16(weights=None)
            except TypeError:
                base_model = tv_models.vgg16(pretrained=False)
            self.nima_model = NIMA(base_model).to(self.device)
            checkpoint = project_path(self.args.nima_ckpt)
            state = torch.load(str(checkpoint), map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.nima_model.load_state_dict(state)
            self.nima_model.eval()
            self.versions["torchvision"] = torchvision.__version__

        if not self.args.skip_lpips:
            import lpips

            self.lpips_model = lpips.LPIPS(net="vgg").to(self.device).eval()
            self.lpips_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
            )
            self.versions["lpips"] = getattr(lpips, "__version__", "unknown")

        if not self.args.skip_fid:
            from torchmetrics.image import FID
            import torchmetrics

            self.fid_class = FID
            self.fid_transform = transforms.Compose(
                [transforms.Resize((299, 299)), transforms.ToTensor()]
            )
            self.versions["torchmetrics"] = torchmetrics.__version__

    def _select_device(self, requested: str):
        if requested == "auto":
            requested = "cuda" if self.torch.cuda.is_available() else "cpu"
        return self.torch.device(requested)

    def clip_image_features(self, path: Path, image: Image.Image):
        key = str(path.resolve())
        if key not in self.image_feature_cache:
            inputs = self.clip_processor(images=image, return_tensors="pt").to(
                self.device
            )
            with self.torch.no_grad():
                features = self.clip_model.get_image_features(**inputs)
            features = self.torch.nn.functional.normalize(features, dim=-1)
            self.image_feature_cache[key] = features.squeeze(0).detach().cpu()
        return self.image_feature_cache[key]

    def clip_text_features(self, text: str):
        if text not in self.text_feature_cache:
            inputs = self.clip_processor(
                text=[text], return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            with self.torch.no_grad():
                features = self.clip_model.get_text_features(**inputs)
            features = self.torch.nn.functional.normalize(features, dim=-1)
            self.text_feature_cache[text] = features.squeeze(0).detach().cpu()
        return self.text_feature_cache[text]

    @staticmethod
    def cosine(features_a, features_b) -> float:
        return float((features_a * features_b).sum().item())

    def lpips(self, image_a: Image.Image, image_b: Image.Image) -> float:
        tensor_a = self.lpips_transform(image_a).unsqueeze(0).to(self.device)
        tensor_b = self.lpips_transform(image_b).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            return float(self.lpips_model(tensor_a, tensor_b).item())

    def _nima_batch(self, image: Image.Image, variant: str):
        T = self.transforms
        normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        if variant == "no_norm_center_crop":
            return T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])(
                image
            ).unsqueeze(0)
        if variant == "center_crop_norm":
            return T.Compose(
                [T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize]
            )(image).unsqueeze(0)
        if variant == "resize_direct_norm":
            return T.Compose([T.Resize((224, 224)), T.ToTensor(), normalize])(
                image
            ).unsqueeze(0)
        if variant == "five_crop_norm":
            crops = T.FiveCrop(224)(T.Resize(256)(image))
            return self.torch.stack([normalize(T.ToTensor()(crop)) for crop in crops])
        raise ValueError("unknown NIMA variant: {}".format(variant))

    def nima(self, path: Path, image: Image.Image, variant: str) -> Dict[str, object]:
        key = (str(path.resolve()), variant)
        if key not in self.nima_cache:
            batch = self._nima_batch(image, variant).to(self.device)
            with self.torch.no_grad():
                distribution = self.nima_model(batch).detach().cpu().numpy()
            if distribution.ndim == 2:
                distribution = distribution.mean(axis=0)
            distribution = distribution.reshape(-1).astype(np.float64)
            total = float(distribution.sum())
            if total <= 0:
                raise ValueError("NIMA returned a non-positive distribution")
            distribution /= total
            scores = np.arange(1, 11, dtype=np.float64)
            mean = float((scores * distribution).sum())
            std = float(np.sqrt(((scores - mean) ** 2 * distribution).sum()))
            self.nima_cache[key] = {
                "mean": mean,
                "std": std,
                "p_low": float(distribution[:4].sum()),
                "p_high": float(distribution[6:].sum()),
                "top_bin": int(distribution.argmax() + 1),
            }
        return dict(self.nima_cache[key])

    def update_fid(
        self,
        pair_label: str,
        method: str,
        category: str,
        generated: Image.Image,
        real: Image.Image,
    ) -> None:
        fake_tensor = (self.fid_transform(generated) * 255).byte().unsqueeze(0)
        real_tensor = (self.fid_transform(real) * 255).byte().unsqueeze(0)
        for group in (category, "ALL"):
            key = (pair_label, method, group)
            if key not in self.fid_metrics:
                self.fid_metrics[key] = self.fid_class(feature=2048).to(self.device)
            metric = self.fid_metrics[key]
            metric.update(real_tensor.to(self.device), real=True)
            metric.update(fake_tensor.to(self.device), real=False)

    def compute_fid(self, pair_label: str, method: str, category: str) -> float:
        key = (pair_label, method, category)
        metric = self.fid_metrics.get(key)
        if metric is None:
            return float("nan")
        try:
            return float(metric.compute().item())
        except Exception as exc:  # FID needs enough samples and a valid backend.
            self.fid_errors[key] = "{}: {}".format(type(exc).__name__, exc)
            return float("nan")


def add_pair_metrics(
    record: Dict[str, object],
    prefix: str,
    generated: Image.Image,
    target: Image.Image,
    runtime: MetricRuntime,
) -> None:
    record.update(prefixed(prefix, paired_pixel_metrics(generated, target)))
    if not runtime.args.skip_lpips:
        record["{}_lpips".format(prefix)] = runtime.lpips(generated, target)


def add_nima_metrics(
    record: Dict[str, object],
    target_label: str,
    path: Path,
    image: Image.Image,
    runtime: MetricRuntime,
) -> None:
    for variant in runtime.args.nima_variants:
        metrics = runtime.nima(path, image, variant)
        record.update(prefixed("nima_{}_{}".format(target_label, variant), metrics))


def append_missing(
    missing: List[Dict[str, object]],
    args: argparse.Namespace,
    row: pd.Series,
    method: str,
    reason: str,
    path: Optional[Path] = None,
) -> None:
    item = {
        "case_number": row.get(args.case_column, ""),
        "image_name": row.get(args.image_column, ""),
        "method": method,
        "reason": reason,
        "path": str(path) if path else "",
    }
    missing.append(item)
    if args.missing_policy == "error":
        raise FileNotFoundError(json.dumps(item, ensure_ascii=False))


def evaluate(
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict], dict]:
    metadata_path = project_path(args.metadata)
    reference_dir = project_path(args.ref_dir)
    runs = parse_runs(args)
    comparison_label = safe_label(args.comparison_label)

    metadata = read_metadata(metadata_path)
    required = [args.image_column, args.prompt_column]
    missing_columns = [column for column in required if column not in metadata.columns]
    if missing_columns:
        raise ValueError(
            "metadata missing required columns: {}".format(missing_columns)
        )

    if args.categories:
        if args.category_column not in metadata.columns:
            raise ValueError(
                "category column not found: {}".format(args.category_column)
            )
        metadata = metadata[
            metadata[args.category_column].astype(str).isin(args.categories)
        ]
    if args.case_numbers:
        if args.case_column not in metadata.columns:
            raise ValueError("case column not found: {}".format(args.case_column))
        requested = {str(value) for value in args.case_numbers}
        metadata = metadata[metadata[args.case_column].astype(str).isin(requested)]
    metadata = metadata.iloc[args.start_index :]
    if args.limit is not None:
        metadata = metadata.head(args.limit)

    runtime = MetricRuntime(args)
    records: List[Dict[str, object]] = []
    missing: List[Dict[str, object]] = []

    total = len(metadata) * len(runs)
    iterator = tqdm(metadata.iterrows(), total=len(metadata), desc="Metadata rows")
    for _, row in iterator:
        image_name = nonempty_text(row[args.image_column])
        prompt = nonempty_text(row[args.prompt_column])
        style_prompt = nonempty_text(row.get(args.style_column, ""))
        category = nonempty_text(row.get(args.category_column, "")) or "unknown"
        reference_path = reference_dir / image_name

        if not reference_path.is_file():
            for method, _ in runs:
                append_missing(
                    missing,
                    args,
                    row,
                    method,
                    "reference image not found",
                    reference_path,
                )
            continue

        reference = load_rgb(reference_path, args.image_size)
        for method, generated_root in runs:
            sample_dir = generated_root / Path(image_name).stem
            generated_path, generated_error = resolve_sample_image(
                sample_dir, args.generated_name, args.glob_policy
            )
            if generated_path is None:
                append_missing(
                    missing,
                    args,
                    row,
                    method,
                    generated_error or "generated image not found",
                    sample_dir,
                )
                continue

            comparison_path, comparison_error = resolve_sample_image(
                sample_dir, args.comparison_name, args.glob_policy
            )
            if args.comparison_name and comparison_path is None:
                append_missing(
                    missing,
                    args,
                    row,
                    method,
                    comparison_error or "comparison image not found",
                    sample_dir,
                )

            generated = load_rgb(generated_path, args.image_size)
            comparison = (
                load_rgb(comparison_path, args.image_size) if comparison_path else None
            )
            record: Dict[str, object] = {
                "case_number": row.get(args.case_column, ""),
                "category": category,
                "image_name": image_name,
                "method": method,
                "prompt": prompt,
                "style_prompt": style_prompt,
                "generated_path": display_path(generated_path),
                "reference_path": display_path(reference_path),
                "comparison_path": display_path(comparison_path)
                if comparison_path
                else "",
            }

            add_pair_metrics(record, "ref", generated, reference, runtime)
            if comparison is not None:
                add_pair_metrics(
                    record, comparison_label, generated, comparison, runtime
                )

            if not args.skip_clip:
                generated_features = runtime.clip_image_features(
                    generated_path, generated
                )
                reference_features = runtime.clip_image_features(
                    reference_path, reference
                )
                prompt_features = runtime.clip_text_features(prompt)
                record["clip_t_prompt"] = runtime.cosine(
                    generated_features, prompt_features
                )
                record["clip_i_ref"] = runtime.cosine(
                    generated_features, reference_features
                )
                if style_prompt:
                    combined_prompt = "{}, {}".format(style_prompt, prompt)
                    record["clip_t_prompt_style"] = runtime.cosine(
                        generated_features, runtime.clip_text_features(combined_prompt)
                    )
                if comparison is not None:
                    comparison_features = runtime.clip_image_features(
                        comparison_path, comparison
                    )
                    record["clip_i_{}".format(comparison_label)] = runtime.cosine(
                        generated_features, comparison_features
                    )
                    record["clip_i_ref_{}".format(comparison_label)] = runtime.cosine(
                        reference_features, comparison_features
                    )
                    comparison_clip_t = runtime.cosine(
                        comparison_features, prompt_features
                    )
                    record["clip_t_{}_prompt".format(comparison_label)] = (
                        comparison_clip_t
                    )
                    record[
                        "clip_t_delta_generated_minus_{}".format(comparison_label)
                    ] = record["clip_t_prompt"] - comparison_clip_t
                    record[
                        "clip_i_ref_delta_generated_minus_{}".format(comparison_label)
                    ] = (
                        record["clip_i_ref"]
                        - record["clip_i_ref_{}".format(comparison_label)]
                    )

            if not args.skip_nima:
                if "generated" in args.nima_targets:
                    add_nima_metrics(
                        record, "generated", generated_path, generated, runtime
                    )
                if "reference" in args.nima_targets:
                    add_nima_metrics(
                        record, "reference", reference_path, reference, runtime
                    )
                if comparison is not None and "comparison" in args.nima_targets:
                    add_nima_metrics(
                        record,
                        comparison_label,
                        comparison_path,
                        comparison,
                        runtime,
                    )
                    for variant in args.nima_variants:
                        for statistic in ("mean", "std", "p_low", "p_high"):
                            generated_key = "nima_generated_{}_{}".format(
                                variant, statistic
                            )
                            comparison_key = "nima_{}_{}_{}".format(
                                comparison_label, variant, statistic
                            )
                            if generated_key in record and comparison_key in record:
                                record[
                                    "nima_{}_{}_delta_generated_minus_{}".format(
                                        variant, statistic, comparison_label
                                    )
                                ] = record[generated_key] - record[comparison_key]

            if not args.skip_fid:
                runtime.update_fid("ref", method, category, generated, reference)
                if comparison is not None:
                    runtime.update_fid(
                        comparison_label,
                        method,
                        category,
                        generated,
                        comparison,
                    )

            records.append(record)

    per_sample = pd.DataFrame(records)
    if per_sample.empty:
        raise RuntimeError(
            "no valid samples were evaluated; inspect paths and missing-file records"
        )
    summary = build_summary(per_sample, runtime, comparison_label)
    log = {
        "task": "DePrism comprehensive image evaluation",
        "metadata": str(metadata_path),
        "reference_dir": str(reference_dir),
        "runs": [{"method": label, "path": str(path)} for label, path in runs],
        "generated_name": args.generated_name,
        "comparison_name": args.comparison_name,
        "comparison_label": comparison_label if args.comparison_name else None,
        "clip_model": None if args.skip_clip else args.clip_model,
        "nima_checkpoint": None if args.skip_nima else args.nima_ckpt,
        "nima_variants": [] if args.skip_nima else args.nima_variants,
        "nima_targets": [] if args.skip_nima else args.nima_targets,
        "device": str(runtime.device) if runtime.device is not None else None,
        "image_size": args.image_size,
        "metadata_rows_after_filter": int(len(metadata)),
        "planned_method_samples": int(total),
        "evaluated_samples": int(len(per_sample)),
        "missing_records": int(len(missing)),
        "versions": runtime.versions,
        "fid_errors": {
            "|".join(key): value for key, value in runtime.fid_errors.items()
        },
    }
    return per_sample, summary, missing, log


def build_summary(
    per_sample: pd.DataFrame,
    runtime: MetricRuntime,
    comparison_label: str,
) -> pd.DataFrame:
    metric_columns = [
        column
        for column in per_sample.columns
        if column.startswith(("ref_", "clip_", "nima_", comparison_label + "_"))
        and pd.api.types.is_numeric_dtype(per_sample[column])
        and not column.endswith("_top_bin")
    ]
    rows: List[Dict[str, object]] = []
    for method in per_sample["method"].drop_duplicates():
        method_data = per_sample[per_sample["method"] == method]
        categories = list(method_data["category"].drop_duplicates()) + ["ALL"]
        for category in categories:
            group = (
                method_data
                if category == "ALL"
                else method_data[method_data["category"] == category]
            )
            row: Dict[str, object] = {
                "method": method,
                "category": category,
                "count": int(len(group)),
            }
            for column in metric_columns:
                values = pd.to_numeric(group[column], errors="coerce")
                row["{}_mean".format(column)] = values.mean()
                row["{}_std".format(column)] = values.std(ddof=1)
            if not runtime.args.skip_fid:
                row["fid_ref"] = runtime.compute_fid("ref", method, category)
                if runtime.args.comparison_name:
                    row["fid_{}".format(comparison_label)] = runtime.compute_fid(
                        comparison_label, method, category
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    summary: pd.DataFrame,
    log: Dict[str, object],
    missing: List[dict],
) -> None:
    lines = [
        "# Comprehensive evaluation report",
        "",
        "- Evaluated samples: `{}`".format(log["evaluated_samples"]),
        "- Missing records: `{}`".format(log["missing_records"]),
        "- Generated image: `{}`".format(log["generated_name"]),
        "- Comparison image: `{}`".format(log["comparison_name"] or "disabled"),
        "- CLIP model: `{}`".format(log["clip_model"] or "disabled"),
        "- NIMA variants: `{}`".format(", ".join(log["nima_variants"]) or "disabled"),
        "",
        "## Metric semantics",
        "",
        "- `ref_*` and `clip_i_ref` compare generated images with raw/reference images.",
        "- Comparison-labelled metrics compare generated images with the optional Standard/baseline image.",
        "- `clip_t_*` and `clip_i_*` are cosine similarities of L2-normalized CLIP embeddings.",
        "- NIMA defaults to ImageNet-normalized five-crop inference (`five_crop_norm`).",
        "- FID is a dataset-level metric and appears only in the summary.",
        "- No legacy single-text softmax CLIP score or subjective weighted score is reported.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
    ]
    if missing:
        lines.extend(
            [
                "",
                "## Missing or ambiguous inputs",
                "",
                "See the generated `*_missing.csv` file for the complete list.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix

    per_sample, summary, missing, log = evaluate(args)
    per_sample_path = output_dir / "{}_per_sample.csv".format(prefix)
    summary_path = output_dir / "{}_summary.csv".format(prefix)
    missing_path = output_dir / "{}_missing.csv".format(prefix)
    report_path = output_dir / "{}_report.md".format(prefix)
    log_path = output_dir / "{}_log.json".format(prefix)

    per_sample.to_csv(per_sample_path, index=False, float_format="%.6f")
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    pd.DataFrame(missing).to_csv(missing_path, index=False)
    log["outputs"] = {
        "per_sample": str(per_sample_path),
        "summary": str(summary_path),
        "missing": str(missing_path),
        "report": str(report_path),
        "log": str(log_path),
    }
    write_report(report_path, summary, log, missing)
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
