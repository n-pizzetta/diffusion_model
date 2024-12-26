"""Test the preprocess.py script."""

import numpy as np
from src.preprocess import preprocess_celeba
import textwrap


def test_preprocess_celeba(tmp_path):
    # Create dummy CSV
    csv_content = textwrap.dedent(
        """\
    image_id,partition
    img_000001.jpg,0
    img_000002.jpg,0
    img_000003.jpg,0
    """
    )
    csv_file = tmp_path / "list_eval_partition.csv"
    csv_file.write_text(csv_content)

    # Create dummy image directory
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Create some dummy images
    from PIL import Image

    for i in range(1, 4):
        img = Image.new("RGB", (128, 128), color=(i * 10, i * 10, i * 10))
        img.save(raw_dir / f"img_{i:06d}.jpg")

    # Prepare output path
    output_dir = tmp_path / "processed"

    # Run the preprocess
    preprocess_celeba(
        input_dir=str(raw_dir),
        output_dir=str(output_dir),
        csv_path=str(csv_file),
        n_images=2,  # Only process 2
        data_type="train",  # Partition=0 is train
        img_size=64,
    )

    # Check output files
    npy_files = list(output_dir.glob("*.npy"))
    assert len(npy_files) == 2, "Expected exactly 2 .npy files to be generated."

    # Optionally verify shape and range
    arr = np.load(npy_files[0])
    # Should have shape = (3, 64, 64) after transforms
    assert arr.shape == (3, 64, 64), "Wrong shape after preprocessing."
    # Values should be in [-1,1]
    assert -1.0 <= arr.min() and arr.max() <= 1.0, "Pixel values not in [-1,1]."
