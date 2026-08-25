# Food-11 Image Classification

Projekat iz predmeta **Veštačka inteligencija sa primenama**.

Tema projekta je klasifikacija slika hrane iz Food-11 skupa podataka korišćenjem neuronskih mreža u PyTorch-u.

## Struktura projekta

```text
Food11-Classification/
│
├── data/
│   ├── training/
│   ├── validation/
│   └── evaluation/
│
├── notebooks/
│   └── food11_analysis.ipynb
│
├── src/
│   ├── data_pipeline.py
│   ├── model.py
│   ├── utils.py
│   ├── train.py
│   ├── experiments.py
│   ├── cross_validation.py
│   ├── compare_cv_results.py
│   ├── train_final.py
│   ├── evaluate_final.py
│   └── plot_learning_curves.py
│
├── results/
├── models/
├── mlruns/
├── .gitignore
├── .gitattributes
└── README.md
```

## Dataset

Korišćen je Food-11 skup podataka sa 11 klasa hrane.

Dataset je podeljen na:

- training
- validation
- evaluation

Dataset se ne čuva u GitHub repozitorijumu zbog veličine.

## Analiza podataka

Analiza skupa podataka urađena je u Jupyter notebook-u:

`notebooks/food11_analysis.ipynb`

U okviru analize urađeno je:

- učitavanje podataka
- provera strukture training, validation i evaluation skupova
- provera postojanja svih klasa
- broj slika po klasama
- osnovna deskriptivna statistika
- prikaz distribucije uzoraka
- provera praznih i nedostajućih klasa
- provera oštećenih i nečitljivih slika
- analiza dimenzija slika
- analiza odnosa širine i visine
- prikaz primera slika iz različitih klasa

Pošto je u pitanju skup slika, klasična analiza NaN vrednosti i korelaciona matrica kao kod tabelarnih podataka nisu direktno primenljive. Umesto toga analizirane su osobine samih slika i raspodela uzoraka po klasama.

## Obrada i transformacije podataka

Učitavanje i transformacije nalaze se u fajlu:

`src/data_pipeline.py`

Slike se skaliraju na dimenziju 224x224 piksela.

Korišćene su sledeće transformacije:

- Resize
- RandomHorizontalFlip
- RandomRotation
- ToTensor
- ImageNet normalizacija

Augmentacija se koristi samo tokom treninga, dok se pri validaciji i evaluaciji ne koristi.

## Modeli

Arhitekture modela definisane su u posebnom modulu:

`src/model.py`

Implementirana su dva modela:

- BasicCNN
- ResNet18

ResNet18 je prilagođen Food-11 problemu tako što je poslednji potpuno povezani sloj promenjen da ima 11 izlaza.

## Reproduktivnost

Za eksperimente je korišćen fiksni random seed:

`42`

Pored toga, beleže se verzije korišćenih biblioteka, CUDA verzija i informacije o GPU-u.

Eksperimenti su izvršavani na:

- NVIDIA RTX A4500
- CUDA 13.2

## Eksperimentalne konfiguracije

Testirano je pet različitih konfiguracija ResNet18 modela.

| Konfiguracija | Optimizer | Learning rate | Augmentacija |
|---|---|---:|---|
| exp_01_resnet_no_aug | Adam | 0.001 | Ne |
| exp_02_resnet_aug | Adam | 0.001 | Da |
| exp_03_resnet_aug_lr_0005 | Adam | 0.0005 | Da |
| exp_04_resnet_aug_adamw | AdamW | 0.0001 | Da |
| exp_05_resnet_aug_sgd | SGD | 0.001 | Da |

Konfiguracije su definisane u:

`src/experiments.py`

## Cross-validation

Za svih pet konfiguracija korišćen je 5-fold cross-validation.

Cross-validation je implementiran u:

`src/cross_validation.py`

Tokom eksperimenata automatski se beleže:

- hiperparametri
- train loss po epohi
- validation loss po epohi
- train accuracy po epohi
- validation accuracy po epohi
- accuracy po fold-u
- precision po fold-u
- recall po fold-u
- F1 po fold-u
- prosečne CV metrike
- standardna devijacija
- vreme treninga po fold-u
- ukupno vreme treninga
- GPU i CUDA informacije

## Poređenje konfiguracija

Rezultati cross-validation eksperimenata nalaze se u:

`results/cross_validation_results.csv`

Grafičko poređenje nalazi se u:

`results/cv_comparison.png`

Najbolji rezultat postigla je konfiguracija:

`exp_04_resnet_aug_adamw`

Njeni parametri su:

- Optimizer: AdamW
- Learning rate: 0.0001
- Augmentacija: True
- Batch size: 64

Prosečna accuracy vrednost kroz 5 foldova iznosila je približno **56.15%**.

## Finalni model

Na osnovu cross-validation rezultata izabrana je najbolja konfiguracija i njom je treniran finalni ResNet18 model.

Model je sačuvan u:

`models/best_food11_resnet18.pth`

Model je dodat u GitHub repozitorijum pomoću Git LFS-a.

## Finalna evaluacija

Finalni model je evaluiran na evaluation skupu.

Dobijeni rezultati:

| Metrika | Rezultat |
|---|---:|
| Accuracy | 68.51% |
| Precision | 70.59% |
| Recall | 68.51% |
| F1 score | 68.66% |
| Inference time | 0.372 ms/slika |
| Model size | 42.73 MB |

## Rezultati

U folderu `results/` nalaze se:

- `cross_validation_results.csv`
- `cv_comparison.png`
- `final_metrics.csv`
- `confusion_matrix.png`
- `roc_curves.png`
- `pr_curves.png`
- `learning_curve_loss.png`
- `learning_curve_accuracy.png`
- `resources.txt`

## Learning curves

Learning curve grafikoni generisani su iz MLflow podataka.

Dostupni su:

- `results/learning_curve_loss.png`
- `results/learning_curve_accuracy.png`

## Confusion matrix

Konfuzioni matriks finalnog modela nalazi se u:

`results/confusion_matrix.png`

## ROC i Precision-Recall krive

Za multiclass problem korišćen je One-vs-Rest pristup.

Rezultati su sačuvani u:

- `results/roc_curves.png`
- `results/pr_curves.png`

## MLflow

Svi eksperimenti su praćeni pomoću MLflow-a.

Logovi se nalaze u folderu:

`mlruns/`

MLflow UI se može pokrenuti komandom:

```bash
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
```

Nakon toga se interfejs otvara na:

`http://localhost:5000`

## Pokretanje skripti

Cross-validation:

```bash
python -m src.cross_validation
```

Poređenje cross-validation rezultata:

```bash
python -m src.compare_cv_results
```

Trening finalnog modela:

```bash
python -m src.train_final
```

Evaluacija finalnog modela:

```bash
python -m src.evaluate_final
```

Generisanje learning curve grafikona:

```bash
python -m src.plot_learning_curves
```

## Git i verzionisanje

Projekat je razvijan kroz više commit-a i eksperimentalni branch:

`experiment/hyperparameter-search`

Na ovom branch-u rađena je optimizacija hiperparametara, cross-validation, poređenje konfiguracija i finalna evaluacija modela.

Finalni model se čuva pomoću Git LFS-a.

## Zaključak

Od pet testiranih konfiguracija najbolji rezultat je postigao ResNet18 model sa AdamW optimizerom, learning rate vrednošću 0.0001 i augmentacijom podataka.

Cross-validation je korišćen za izbor najbolje konfiguracije, dok je evaluation skup korišćen za završnu proveru finalnog modela.
