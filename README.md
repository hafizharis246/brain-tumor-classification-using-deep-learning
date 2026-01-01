# brain-tumor-classification-using-deep-learning

# Brain Tumor Detection using Transfer Learning

## Project state

This repository contains the Colab/Jupyter notebook used to build and evaluate a brain tumor detection model. 

Files in this repo:

- `brain_tumor_detection_using_deep_learning.ipynb` — primary Colab/Jupyter notebook (training, evaluation, and inference examples)
- `README.md` — this file

## Notebook summary (what it does)

- Mounts Google Drive to access dataset and save artifacts:

```python
from google.colab import drive
drive.mount('/content/drive')
```

- Loads dataset from Drive paths used in the notebook:

```
train_dir = '/content/drive/MyDrive/Datasets/brain-tumor-classification-dataset/Training/'
test_dir  = '/content/drive/MyDrive/Datasets/brain-tumor-classification-dataset/Testing/'
```

- Data loading and preprocessing
  - Collects file paths and labels from `Training/` and `Testing/` folders and shuffles them.
  - Uses helper functions to open images, resize to `IMAGE_SIZE = 128`, apply simple augmentation (random brightness and contrast), normalize pixels to [0,1], and encode labels.
  - Batching is performed via a custom `datagen(paths, labels, batch_size, epochs)` generator.

- Model architecture
  - Uses `VGG16` (ImageNet weights) with `include_top=False` as the base feature extractor.
  - Freezes all base layers then unfreezes the last 3 layers for fine-tuning.
  - Adds a head: `Flatten` → `Dropout(0.3)` → `Dense(128, relu)` → `Dropout(0.2)` → `Dense(num_classes, softmax)`.
  - Compiles with `Adam(learning_rate=0.0001)`, `loss='sparse_categorical_crossentropy'`, and `metrics=['sparse_categorical_accuracy']`.
  - Training params in the notebook: `batch_size = 20`, `epochs = 5` (adjust inside the notebook as needed).

#+ Evaluation and analysis
  - Plots training history (accuracy & loss).
  - Generates classification report and confusion matrix using scikit-learn on test set predictions.
  - Computes multi-class ROC curves and AUC.

#+ Save / load model
  - Saves model with `model.save('model.h5')` and demonstrates loading via `load_model('model.h5')`.

#+ Inference helper
  - Notebook includes `detect_and_display(img_path, model, image_size=128)` which:
    - Loads and preprocesses a single image
    - Runs `model.predict` and maps predicted index to a human-readable label
    - Displays the image with predicted label and confidence
  - The notebook defines a `class_labels` list used in inference:

```
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']
```

  Note: the mapping between encoded labels (from `os.listdir(train_dir)`) and `class_labels` must match when you train or use the model. If the folder ordering differs, update `class_labels` or the encoding function for consistency.

## Dataset

- Expected dataset layout (as used in the notebook):

```
/content/drive/MyDrive/Datasets/brain-tumor-classification-dataset/
  ├─ Training/
  │   ├─ pituitary/
  │   ├─ glioma/
  │   ├─ notumor/
  │   └─ meningioma/
  └─ Testing/
      ├─ pituitary/
      ├─ glioma/
      ├─ notumor/
      └─ meningioma/
```

#+ You can obtain similar datasets from Kaggle (search for "brain tumor classification" or related). In Colab either upload the extracted dataset to Google Drive or use the Kaggle API to download directly into the runtime, then update `train_dir`/`test_dir` accordingly.

## How to run (recommended: Google Colab)

1. Open `brain_tumor_detection_using_deep_learning.ipynb` in Google Colab.
2. Mount Google Drive (run the mount cell).
3. Verify and update `train_dir` and `test_dir` paths to your Drive dataset location.
4. Install required libraries in a Colab cell if not already present:

```bash
pip install tensorflow scikit-learn matplotlib seaborn pillow
```

5. Run cells sequentially: data loading → preprocessing → model build → training → evaluation.
6. After training, save the model (the notebook uses `model.save('model.h5')`). You can then copy the saved `model.h5` to your Drive and download a copy to this repo if desired.



## Notes

- The notebook contains the full training pipeline using a VGG16 transfer learning setup (feature extractor + custom classifier head). Adjust hyperparameters inside the notebook as needed.
- If you intend to re-run training in Colab, ensure GPU runtime is enabled: Runtime → Change runtime type → Hardware accelerator → GPU.

## Reproducing locally

You can run the notebook locally with Jupyter, but large training is best performed in Colab or a GPU-enabled environment. If running locally, install dependencies used in the notebook (see earlier `pip` command).


