# Food-11 Image Classification

Projekat iz predmeta **Veštačka inteligencija sa primenama**.

Projekat obuhvata kompletan sistem za klasifikaciju slika hrane iz Food-11 skupa podataka. Sistem uključuje analizu podataka, preprocessing pipeline, treniranje i evaluaciju ResNet18 modela, praćenje eksperimenata pomoću MLflow-a, Streamlit web aplikaciju za interakciju sa modelom i Docker kontejner za pokretanje aplikacije.

## Struktura projekta

```text
Food11-Classification/
├── app/
│   ├── __init__.py
│   ├── app.py
│   └── inference.py
├── data/
│   ├── training/
│   ├── validation/
│   └── evaluation/
├── mlruns/
├── models/
│   └── best_food11_resnet18.pth
├── notebooks/
│   └── food11_analysis.ipynb
├── results/
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
├── .dockerignore
├── .gitattributes
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

## Dataset

Korišćen je Food-11 skup podataka za klasifikaciju slika hrane sa 11 klasa:

- bread
- dairy_product
- dessert
- egg
- fried_food
- meat
- noodles_pasta
- rice
- seafood
- soup
- vegetable_fruit

Dataset je podeljen na training, validation i evaluation skupove. Zbog veličine se ne čuva u GitHub repozitorijumu.

## Analiza podataka

Eksploratorna analiza urađena je u `notebooks/food11_analysis.ipynb`.

Analiza obuhvata:

- proveru strukture skupa
- broj slika po klasama
- deskriptivnu statistiku
- distribuciju uzoraka
- proveru praznih i nedostajućih klasa
- proveru oštećenih i nečitljivih slika
- analizu dimenzija i odnosa širine i visine
- prikaz primera slika

## Preprocessing pipeline

Preprocessing je implementiran u `src/data_pipeline.py`.

Slike se skaliraju na 224 × 224 piksela.

Tokom treninga koriste se Resize, RandomHorizontalFlip, RandomRotation, ToTensor i ImageNet normalizacija. Tokom validacije, evaluacije i inferencije ne koristi se augmentacija.

## Model

Arhitekture se nalaze u `src/model.py`.

Implementirani su:

- BasicCNN
- ResNet18

Za finalni sistem korišćen je ResNet18 sa 11 izlaznih klasa.

## Reproduktivnost

Za eksperimente je korišćen fiksni random seed `42`.

Eksperimenti su izvršavani na NVIDIA RTX A4500 GPU-u, uz evidentiranje verzija biblioteka, CUDA okruženja i hardverskih informacija.

## Eksperimentalne konfiguracije

| Konfiguracija | Optimizer | Learning rate | Augmentacija |
|---|---|---:|---|
| exp_01_resnet_no_aug | Adam | 0.001 | Ne |
| exp_02_resnet_aug | Adam | 0.001 | Da |
| exp_03_resnet_aug_lr_0005 | Adam | 0.0005 | Da |
| exp_04_resnet_aug_adamw | AdamW | 0.0001 | Da |
| exp_05_resnet_aug_sgd | SGD | 0.001 | Da |

Konfiguracije su definisane u `src/experiments.py`.

## Cross-validation

Za svih pet konfiguracija korišćen je 5-fold cross-validation u `src/cross_validation.py`.

Beleženi su hiperparametri, train/validation loss i accuracy po epohi, metrike po fold-u, prosečne metrike, standardna devijacija, vreme treninga i hardverske informacije.

## Poređenje konfiguracija

Rezultati se nalaze u:

- `results/cross_validation_results.csv`
- `results/cv_comparison.png`

Najbolja konfiguracija je `exp_04_resnet_aug_adamw`:

- optimizer: AdamW
- learning rate: 0.0001
- augmentacija: uključena
- batch size: 64

Prosečna accuracy vrednost kroz 5 foldova iznosila je približno **56.15%**.

## Finalni model

Finalni model se nalazi u:

`models/best_food11_resnet18.pth`

Model se čuva pomoću Git LFS-a.

## Finalna evaluacija

| Metrika | Rezultat |
|---|---:|
| Accuracy | 68.51% |
| Precision | 70.59% |
| Recall | 68.51% |
| F1 score | 68.66% |
| Inference time | 0.372 ms/slika |
| Model size | 42.73 MB |

## Rezultati

U `results/` nalaze se:

- `cross_validation_results.csv`
- `cv_comparison.png`
- `final_metrics.csv`
- `confusion_matrix.png`
- `roc_curves.png`
- `pr_curves.png`
- `learning_curve_loss.png`
- `learning_curve_accuracy.png`
- `resources.txt`

## MLflow

MLflow logovi nalaze se u `mlruns/`.

Pokretanje MLflow UI-ja:

```bash
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
```

Interfejs je zatim dostupan na `http://localhost:5000`.

## Streamlit aplikacija

Front-end je implementiran pomoću Streamlit biblioteke.

Glavni fajl:

`app/app.py`

Inference logika:

`app/inference.py`

Aplikacija omogućava:

- upload JPG/JPEG/PNG slike
- prikaz učitane slike
- klasifikaciju treniranim ResNet18 modelom
- prikaz najverovatnije klase
- confidence vrednost
- top 3 predikcije
- inference vreme

Model je namenjen slikama iz Food-11 domena. Za slike van domena može dati pogrešnu predikciju sa visokom softmax verovatnoćom.

## Lokalno pokretanje aplikacije

Instalacija zavisnosti:

```bash
pip install -r requirements.txt
```

Pokretanje:

```bash
streamlit run app/app.py
```

Aplikacija je dostupna na `http://localhost:8501`.

## Docker

Docker image se pravi iz root foldera projekta:

```bash
docker build -t food11-classifier .
```

Kontejner se pokreće:

```bash
docker run --rm -p 8501:8501 food11-classifier
```

Aplikacija je zatim dostupna na `http://localhost:8501`.

Docker image sadrži Python okruženje, PyTorch, torchvision, Streamlit, kod aplikacije, kod modela i trenirani model.

`.dockerignore` sprečava uključivanje nepotrebnih fajlova kao što su dataset, notebook fajlovi, rezultati i MLflow logovi u Docker build context.

## Pokretanje ML eksperimenata

Cross-validation:

```bash
python -m src.cross_validation
```

Poređenje CV rezultata:

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

Learning curves:

```bash
python -m src.plot_learning_curves
```

## Git i verzionisanje

Eksperimenti sa hiperparametrima razvijani su na branch-u:

`experiment/hyperparameter-search`

Proširenje projekta za seminarski rad razvijano je na branch-u:

`seminarski`

Git istorija prikazuje razvoj od analize i preprocessing-a, preko eksperimentisanja i finalnog modela, do Streamlit aplikacije i Docker implementacije.

## Zaključak

Razvijen je kompletan sistem za klasifikaciju slika hrane zasnovan na ResNet18 neuronskoj mreži.

Cross-validation je korišćen za izbor najbolje konfiguracije, finalni model je evaluiran na zasebnom evaluation skupu, a sistem je proširen Streamlit web aplikacijom i Docker kontejnerom za reproduktivno pokretanje.
