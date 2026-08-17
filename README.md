# Rice Leaf Disease Classification

A small transfer-learning experiment for classifying three rice leaf conditions from photographs. The notebook was developed as an academic exercise using PyTorch and a pre-trained ResNet-18 model.

## Scope

The source dataset contains 120 images divided equally between:

- Bacterial leaf blight
- Brown spot
- Leaf smut

The notebook explores image augmentation, transfer learning, confusion-matrix analysis and qualitative review of correct and incorrect predictions.

Because the dataset is small and the saved notebook does not preserve a fixed independent test split, the results should be treated as preliminary. They are not a reliable estimate of performance in field conditions.

## Workflow

```text
Kaggle image dataset
    -> resize and augmentation
    -> ImageNet normalisation
    -> pre-trained ResNet-18
    -> three-class output layer
    -> confusion matrix and per-class evaluation
```

## Running the notebook

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/SidhanthChavan/Rice-Leaf-Disease-Detection.git
cd Rice-Leaf-Disease-Detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook rice_leaf_detection.ipynb
```

The notebook downloads the `vbookshelf/rice-leaf-diseases` dataset through `kagglehub`. Kaggle credentials may be required in a local environment. Some cells also contain Kaggle-specific paths and must be adjusted when running elsewhere.

## Current limitations

- Only 120 source images are available.
- The notebook creates validation selections more than once without preserving fixed indices.
- No independent, untouched test set is stored.
- Some later plotting cells use illustrative history values rather than a persisted training log.
- No trained model or deployment package is included.

A credible follow-up would use a larger dataset, a stratified split saved to disk, seeded experiments and evaluation on images collected from a different source.

## Stack

Python, PyTorch, torchvision, ResNet-18, scikit-learn, OpenCV, NumPy, Pandas, Matplotlib and Seaborn.

## Licence

See [LICENSE](LICENSE).
