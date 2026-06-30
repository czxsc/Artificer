from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter


CORRUPTION_RECIPES: dict[str, dict[str, object]] = {
    "gaussian_blur": {"radius": 2.0},
    "strong_gaussian_blur": {"radius": 4.0},
    "small_center_crop": {"crop_ratio": 0.90},
    "medium_center_crop": {"crop_ratio": 0.75},
    "warm_shift": {"color_factor": 1.25, "brightness_factor": 1.03},
    "desaturated": {"color_factor": 0.55, "brightness_factor": 1.0},
    "tilt_left": {"angle": -8},
    "tilt_right": {"angle": 8},
    "perspective_left": {"shear": -0.12},
    "perspective_right": {"shear": 0.12},
}


def apply_corruption(image: Image.Image, corruption_name: str) -> Image.Image:
    if corruption_name not in CORRUPTION_RECIPES:
        raise ValueError(f"Unknown corruption: {corruption_name}")

    recipe = CORRUPTION_RECIPES[corruption_name]
    image = image.convert("RGB")

    if corruption_name in {"gaussian_blur", "strong_gaussian_blur"}:
        return image.filter(ImageFilter.GaussianBlur(radius=float(recipe["radius"])))

    if corruption_name in {"small_center_crop", "medium_center_crop"}:
        crop_ratio = float(recipe["crop_ratio"])
        width, height = image.size
        crop_width = int(width * crop_ratio)
        crop_height = int(height * crop_ratio)
        left = max(0, (width - crop_width) // 2)
        top = max(0, (height - crop_height) // 2)
        cropped = image.crop((left, top, left + crop_width, top + crop_height))
        return cropped.resize((width, height), Image.Resampling.BICUBIC)

    if corruption_name in {"warm_shift", "desaturated"}:
        color = ImageEnhance.Color(image).enhance(float(recipe["color_factor"]))
        return ImageEnhance.Brightness(color).enhance(float(recipe["brightness_factor"]))

    if corruption_name in {"tilt_left", "tilt_right"}:
        return image.rotate(
            float(recipe["angle"]),
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=(245, 245, 245),
        )

    if corruption_name in {"perspective_left", "perspective_right"}:
        width, height = image.size
        shear = float(recipe["shear"])
        shift = abs(shear) * width
        new_width = int(width + shift)
        transformed = image.transform(
            (new_width, height),
            Image.Transform.AFFINE,
            (1, shear, -shift if shear > 0 else 0, 0, 1, 0),
            resample=Image.Resampling.BICUBIC,
            fillcolor=(245, 245, 245),
        )
        return transformed.resize((width, height), Image.Resampling.BICUBIC)

    raise ValueError(f"Unhandled corruption: {corruption_name}")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
