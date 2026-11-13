# TCAV with ImageNette and DTD

## Team Members

**Course Project -- Interpretable Machine Learning (IML)**

  Name                           Roll Number
  ------------------------------ -------------
  Pareddi Sai Dinesh Reddy       24AI10041
  Metla Gowtham Venkata Mahesh   24AI10037
  Bokka Navadeep Pavan           24AI10019
  Sabavath Arun Singh            24AI10005
  Bhukya Saidulu                 24AI10002

------------------------------------------------------------------------

## 1. Project Overview

This project implements **Testing with Concept Activation Vectors
(TCAV)** to investigate whether a deep learning image classifier uses
human-understandable visual concepts when making predictions.

Traditional feature-attribution methods explain model predictions using
individual pixels or regions. TCAV instead measures the influence of
high-level concepts, such as textures, on a model's predictions.

We use a ResNet50 image classifier fine-tuned on ImageNette and evaluate
five texture concepts from the Describable Textures Dataset (DTD).

### Objectives

-   Implement the TCAV methodology using a fine-tuned ResNet50 model.
-   Learn Concept Activation Vectors (CAVs) for selected texture
    concepts.
-   Calculate TCAV scores for ImageNette target classes.
-   Compare true-concept scores against random-reference and
    shuffled-label controls.
-   Statistically evaluate the predefined concept--class hypotheses.
-   Visualize concept influence across different model layers.

------------------------------------------------------------------------

## 2. Paper Reference

**Kim et al.Â (2018).**

*Interpretability Beyond Feature Attribution: Quantitative Testing with
Concept Activation Vectors (TCAV).*

Proceedings of the 35th International Conference on Machine Learning
(ICML).

The implementation follows the central TCAV idea of quantifying the
sensitivity of a model's prediction to a human-defined concept direction
in its learned representation space.

------------------------------------------------------------------------

## 3. Datasets

### 3.1 ImageNette

ImageNette is a subset of ImageNet containing ten image classes. It is
used to train and evaluate the target image classifier.

-   Training dataset: Official ImageNette training split.
-   Validation dataset: 10% stratified split from the official training
    data.
-   Test dataset: Official ImageNette validation split.
-   Input image size: 224 Ã— 224 pixels.
-   Resize dimension: 256 pixels.

The ten target classes include:

1.  Tench
2.  English springer
3.  Cassette player
4.  Chain saw
5.  Church
6.  French horn
7.  Garbage truck
8.  Gas pump
9.  Golf ball
10. Parachute

### 3.2 Describable Textures Dataset (DTD)

DTD is used to obtain human-understandable texture concepts.

Five texture categories are selected:

-   Dotted
-   Striped
-   Banded
-   Zigzagged
-   Woven

For each concept:

-   120 images are sampled.
-   A reference pool of 1,500 images is constructed using other DTD
    texture categories.
-   Additional visually related categories are excluded from the
    reference pool where specified in the configuration.

The texture images are used to train concept classifiers and construct
CAVs.

### Dataset Sources

-   ImageNette: https://github.com/fastai/imagenette
-   DTD: https://www.robots.ox.ac.uk/~vgg/data/dtd/

The implementation automatically downloads the datasets when they are
not already available in the configured data directory.

------------------------------------------------------------------------

## 4. Experimental Design

### 4.1 Target Model

A ResNet50 architecture is used as the target image classifier.

-   Architecture: ResNet50
-   Initial weights: ImageNet-1K V2 pretrained weights
-   Number of target classes: 10
-   Input resolution: 224 Ã— 224
-   Optimizer: AdamW
-   Learning rate: 0.0001
-   Weight decay: 0.0001
-   Batch size: 64
-   Training epochs: 6
-   Random seed: 42

The final model checkpoint is selected using validation performance.

### 4.2 Concept--Class Hypotheses

Five hypotheses are defined before conducting the confirmatory analysis.

  Texture Concept   Target ImageNette Class
  ----------------- -------------------------
  Dotted            Golf ball
  Striped           English springer
  Banded            Parachute
  Zigzagged         Chain saw
  Woven             Cassette player

These hypotheses are evaluated using TCAV scores and compared against
random-reference controls.

### 4.3 Layers Evaluated

TCAV analysis is conducted at two ResNet50 representation layers:

  Layer      Feature Dimension
  -------- -------------------
  layer3                  1024
  layer4                  2048

Spatial feature maps are pooled to obtain feature representations for
CAV learning.

**layer3 is the primary layer** for interpreting the main experimental
findings.

------------------------------------------------------------------------

## 5. Methodology

The overall pipeline consists of the following stages:

1.  Dataset preparation and preprocessing.
2.  Training the target ResNet50 classifier.
3.  Extracting intermediate layer activations.
4.  Constructing concept and reference feature datasets.
5.  Training concept classifiers.
6.  Obtaining Concept Activation Vectors.
7.  Computing directional derivatives.
8.  Calculating TCAV scores.
9.  Running control experiments.
10. Performing statistical analysis and visualization.

### 5.1 Concept Activation Vectors

A Concept Activation Vector represents a human-defined concept as a
direction in the model's activation space.

For each concept, a binary classifier is trained to distinguish:

-   Positive examples: Images belonging to the selected texture concept.
-   Negative examples: Images from the reference pool.

A linear classifier trained with logistic loss and L2 regularization is
used to obtain the concept direction.

The feature representations are centered before classifier training. The
resulting CAV is normalized and oriented toward increasing concept
presence.

### 5.2 TCAV Score

The directional derivative measures how sensitive a class prediction is
to movement in the direction of a concept vector.

For a class (k), concept direction (v_C), and activation representation
(f_l(x)), the directional derivative is:

\[ S\_{C,k}(x) = `\nabla`{=tex}\_{f_l(x)}h_k(f_l(x)) `\cdot `{=tex}v_C
\]

The TCAV score is the fraction of evaluated examples for which this
derivative is positive:

\[ TCAV\_{C,k} = `\frac{
|\{x : S_{C,k}(x)>0\}|
}{
|X_k|
}`{=tex} \]

A higher TCAV score indicates that increasing the corresponding concept
direction tends to increase the class score locally.

### 5.3 Layer-wise Intervention

For layer3, the implementation uses a spatially uniform shift of the
activation map corresponding to the pooled CAV direction. This allows
the directional derivative to be evaluated with respect to the layer3
representation.

For layer4, pooled features feed a linear classification head.
Consequently, the gradient with respect to the pooled representation is
image-independent for a given class. This can produce degenerate TCAV
scores of 0 or 1.

Therefore, layer3 is used as the primary layer for interpreting the
concept results.

### 5.4 Control Experiments

Two control approaches are used.

**1. Shuffled-label control**

Concept labels are shuffled before training the concept classifier. This
tests whether the observed results depend on meaningful concept labels.

**2. Random-reference control**

The random-reference control constructs CAVs using two randomly sampled
subsets of the reference pool. One subset is assigned positive labels
and the other negative labels.

Unlike the true-concept CAVs, this control does not use the selected
concept images. It provides a baseline for measuring the influence of
directions learned from arbitrary distinctions within the reference
pool.

The true-concept TCAV scores are compared against the TCAV scores
obtained from these random-reference CAVs.

### 5.5 Experimental Repetitions

The full experiment uses:

-   30 true-concept CAV runs per concept and layer.
-   50 shuffled-label control runs per concept and layer.
-   50 random-reference runs per layer.

Repeated runs are used to evaluate the robustness of the learned concept
directions and their resulting TCAV scores. They measure variability
from CAV sampling and training, not independent dataset-level
replication.

------------------------------------------------------------------------

## 6. Statistical Analysis

True-concept TCAV scores are compared against the corresponding control
scores.

### Statistical Tests

  -----------------------------------------------------------------------
  Analysis                            Method
  ----------------------------------- -----------------------------------
  Primary test                        Two-sided Mann--Whitney U test on
                                      TCAV scores

  Secondary test                      Welch's independent two-sample
                                      t-test

  Additional secondary test           Mann--Whitney U test on mean
                                      directional derivatives

  Effect size                         Difference in mean TCAV scores

  Effect size                         Cliff's delta
  -----------------------------------------------------------------------

### Multiple-Comparison Correction

Two correction methods are applied:

-   Holm correction
-   Benjamini--Hochberg false discovery rate correction

Corrections are performed separately within each layer and control
family.

Two types of analysis are reported:

**Confirmatory analysis:** Evaluates only the five predefined
concept--class hypotheses.

**Exploratory analysis:** Evaluates all concept--class pairs.

A hypothesis is considered statistically supported in the confirmatory
analysis when its Holm-adjusted primary Mann--Whitney U p-value is below
0.05.

------------------------------------------------------------------------

## 7. Experimental Results

### 7.1 Target Model Performance

The final ResNet50 checkpoint completed six training epochs.

  Metric                   Result
  --------------------- ---------
  Training accuracy        99.53%
  Validation accuracy      99.79%
  Validation loss         0.00740
  Test accuracy            99.24%
  Test loss               0.02205

The model achieved approximately 99.24% accuracy on the official
ImageNette validation split used as the test set.

### 7.2 Confirmatory TCAV Results --- layer3

The following table reports the true-concept TCAV scores compared
against random-reference controls.

  ---------------------------------------------------------------------------------
  Hypothesis      Mean TCAV Mean Control   Difference   Holm-adjusted Conclusion
                                                              p-value 
  ------------ ------------ ------------ ------------ --------------- -------------
  Dotted â†’            0.889        0.537       +0.352    8.11 Ã— 10â»Â¹Â² Supported
  Golf ball                                                           

  Striped â†’           0.392        0.462       âˆ’0.071           0.650 Not
  English                                                             significant
  springer                                                            

  Banded â†’            0.822        0.476       +0.346    4.99 Ã— 10â»Â¹Â² Supported
  Parachute                                                           

  Zigzagged â†’         0.413        0.501       âˆ’0.087           0.158 Not
  Chain saw                                                           significant

  Woven â†’             0.417        0.477       âˆ’0.061           0.650 Not
  Cassette                                                            significant
  player                                                              
  ---------------------------------------------------------------------------------

### 7.3 Interpretation

Two of the five predefined hypotheses are statistically supported
against the random-reference control after Holm correction.

**Dotted â†’ Golf ball**

The dotted concept achieves a mean TCAV score of approximately 0.889,
compared with 0.537 for the random-reference control. The difference is
statistically significant after correction.

**Banded â†’ Parachute**

The banded concept achieves a mean TCAV score of approximately 0.822,
compared with 0.476 for the random-reference control. This hypothesis is
also statistically significant after correction.

**Other hypotheses**

The striped, zigzagged, and woven concepts do not show statistically
significant differences from their random-reference controls under the
confirmatory criterion.

Although their observed mean TCAV scores are lower than their control
means, these differences should not be interpreted as statistically
established negative relationships.

### 7.4 Visualizations

The experiment generates visualizations to inspect concept influence and
model behavior.

The main figures include:

-   Training and validation curves.
-   Confusion matrix.
-   CAV validation accuracy.
-   TCAV heatmaps for layer3 and layer4.
-   Hypothesis-versus-random-reference comparisons.
-   Concept influence across ImageNette classes.
-   Concept influence for individual target classes.

Figures are available in:

`results/figures/`

------------------------------------------------------------------------

## 8. Repository Structure

```text
TCAV-ImageNette-DTD/
|
|-- config.json
|-- configs/
|   `-- default.json
|
|-- notebooks/
|   `-- imlnotebook.ipynb
|
|-- report/
|   `-- report.pdf
|
|-- results/
|   |-- cav_runs.csv
|   |-- stats_confirmatory_hypotheses.csv
|   |-- stats_exploratory_all_pairs.csv
|   |-- tcav_results_per_run.csv
|   |-- test_metrics.json
|   |-- test_predictions.csv
|   |-- training_history.json
|   `-- figures/
|
|-- scripts/
|   |-- train_model.py
|   |-- extract_features.py
|   |-- run_tcav.py
|   |-- analyze_results.py
|   `-- package_outputs.py
|
|-- src/
|   `-- tcav_imagenette/
|
|-- tests/
|
|-- requirements.txt
|-- pyproject.toml
|-- .gitignore
`-- README.md
```

`notebooks/imlnotebook.ipynb` contains the Colab-oriented notebook
implementation. The modular implementation is organized into `src/` and
`scripts/`.

The root `config.json` contains the configuration used by the Colab
notebook, including its Colab paths and Google Drive settings.
`configs/default.json` is the configuration for the modular Python
scripts and uses relative local paths.

Downloaded datasets, model checkpoints, cached feature tensors, and
temporary experiment backups are not included in the tracked repository
structure.

------------------------------------------------------------------------

## 9. Installation and Execution

### Requirements

-   Python 3.9 or later
-   PyTorch 2.0 or later
-   Torchvision 0.15 or later
-   Other dependencies listed in `pyproject.toml` and `requirements.txt`

### 9.1 Clone the Repository

``` bash
git clone https://github.com/saidinesh1904/TCAV-ImageNette-DTD.git
cd TCAV-ImageNette-DTD
```

### 9.2 Create a Virtual Environment

**Windows PowerShell:**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

``` bash
python -m venv .venv
source .venv/bin/activate
```

### 9.3 Install Dependencies

``` bash
pip install --upgrade pip
pip install -e .
```

Alternatively:

``` bash
pip install -r requirements.txt
```

### 9.4 Run the Modular Python Implementation

The modular implementation separates model training, feature extraction,
TCAV execution, and statistical analysis into individual scripts.

Run the scripts from the repository root, passing the project
configuration explicitly:

``` bash
python scripts/train_model.py --config configs/default.json
python scripts/extract_features.py --config configs/default.json
python scripts/run_tcav.py --config configs/default.json
python scripts/analyze_results.py --config configs/default.json --plots
```

The commands run the full experiment using the settings in
`configs/default.json`. The first three stages generate or reuse the
required datasets, model checkpoint, cached features, and TCAV results.
The analysis stage reads the generated results and writes statistical
summaries and figures.

To inspect available options:

``` bash
python scripts/train_model.py --help
python scripts/extract_features.py --help
python scripts/run_tcav.py --help
python scripts/analyze_results.py --help
```

The default experiment configuration is stored in:

`configs/default.json`

Dataset downloads and pretrained model weights are handled automatically
when required and when they are not already cached.

**Note:** The complete experiment is computationally intensive. Google
Colab with GPU acceleration is recommended for reproducing the full
experiment. CPU execution is possible but can take considerably longer.

### 9.5 Run the Notebook in Google Colab

The notebook is available at:

`notebooks/imlnotebook.ipynb`

It is designed for Google Colab and includes notebook-specific setup,
dataset handling, training, feature extraction, TCAV experiments,
statistical analysis, and visualizations.

Open the notebook in Google Colab and run its cells in order. The
notebook configuration uses `/content` paths and can optionally save
outputs to Google Drive. These notebook paths are separate from the
relative paths used by the modular Python configuration.

------------------------------------------------------------------------

## 10. Testing

The project includes unit tests covering:

-   Configuration loading and validation.
-   Dataset handling and caching.
-   Model initialization and data splitting.
-   CAV construction.
-   TCAV computation.
-   Statistical analysis.

Run the tests using:

``` bash
pytest -v
```

The test suite completed with **46 passing tests**.

------------------------------------------------------------------------

## 11. Reproducibility

The experiment uses a fixed random seed:

``` text
SEED = 42
```

Important configuration parameters include:

  Parameter                           Value
  -------------------------------- --------
  Image size                            224
  Resize                                256
  Batch size                             64
  Training epochs                         6
  Learning rate                      0.0001
  Weight decay                       0.0001
  Concept images per category           120
  Reference pool size                  1500
  True-concept CAV runs                  30
  Control runs                           50
  Primary layer                      layer3
  Statistical significance level       0.05

The complete configuration used for the modular experiment is stored in
`configs/default.json`. The Colab notebook configuration is stored
separately in the root `config.json`.

------------------------------------------------------------------------

## 12. Limitations

-   The experiment uses a limited set of five texture concepts and ten
    ImageNette classes.
-   DTD texture images and ImageNette object images have different
    visual characteristics and image distributions.
-   The predefined concept--class relationships are hypotheses to be
    tested, not guaranteed relationships.
-   Repeated CAV runs evaluate robustness to sampling and
    concept-classifier training variation. They do not represent
    independent dataset-level replications.
-   The layer4 representation feeds a linear classification head, which
    can lead to degenerate directional-derivative results. The layer3
    results are therefore emphasized.
-   TCAV measures local sensitivity along a concept direction; it does
    not establish that a concept is the sole reason for a model
    prediction or prove causal dependence.

------------------------------------------------------------------------

## 13. Conclusion

This project implements TCAV to investigate the influence of
human-understandable texture concepts on a ResNet50 image classifier
trained on ImageNette.

The model achieved 99.24% test accuracy. The confirmatory layer3
analysis found statistically significant evidence for two predefined
relationships:

-   Dotted texture â†’ Golf ball
-   Banded texture â†’ Parachute

The remaining three hypotheses were not statistically supported against
the random-reference control after Holm correction.

The results demonstrate how concept-based interpretability can be used
to quantitatively investigate the internal representations of image
classification models, while also highlighting the importance of
controls, statistical testing, and careful interpretation.

------------------------------------------------------------------------

## 14. References

1.  Kim, B., et al.Â (2018). *Interpretability Beyond Feature
    Attribution: Quantitative Testing with Concept Activation Vectors
    (TCAV).* Proceedings of the 35th International Conference on Machine
    Learning (ICML).
2.  ImageNette Dataset: https://github.com/fastai/imagenette
3.  Describable Textures Dataset (DTD):
    https://www.robots.ox.ac.uk/~vgg/data/dtd/
4.  PyTorch: https://pytorch.org/
5.  Torchvision: https://pytorch.org/vision/

------------------------------------------------------------------------

## Repository

GitHub: https://github.com/saidinesh1904/TCAV-ImageNette-DTD

