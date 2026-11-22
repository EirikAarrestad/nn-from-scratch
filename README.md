# Nevralt nettverk fra bunnen av

Et nevralt nettverk skrevet fra bunnen av i Python, med bare NumPy, som kjenner igjen håndskrevne siffer fra MNIST-datasettet. Det er ikke brukt noe rammeverk for dyp læring: forward-pass, backpropagation, tapsfunksjon og optimering er implementert for hånd.

## Resultat

Nettverket klassifiserer **97,6 %** av bildene i testsettet riktig etter 10 epoker. Dette er 10 000 bilder modellen aldri har sett under treningen.

| Arkitektur                      | Optimering                | Epoker | Treffsikkerhet (test) |
| ------------------------------- | ------------------------- | ------ | --------------------- |
| 784 → 128 (ReLU) → 10 (Softmax) | SGD, lr = 0.1, batch = 64 | 10     | 97,6 %                |

Tapet sank jevnt fra 0.36 etter første epoke til 0.056 etter tiende.

## Fremgangsmåte

Hvert bilde er 28 × 28 piksler i gråtoner. Bildet flates ut til en vektor med 784 verdier, skalert til [0, 1], og nettverket skal plassere det i én av 10 klasser (sifrene 0–9).

- **Dense-lag:** regner ut `XW + b`. Vektene initialiseres med He-initialisering.
- **Aktivering:** ReLU i det skjulte laget, og Softmax i output-laget. Softmax trekker fra maksverdien i hver rad for å unngå overflow.
- **Tapsfunksjon:** cross-entropy. Når Softmax og cross-entropy kombineres, blir gradienten `(y_pred − y_true) / batchstørrelse`.
- **Optimering:** SGD med mini-batcher. Treningsdataene stokkes på nytt i hver epoke.

## Gradient-sjekk

For å kontrollere at backpropagation er riktig, sammenligner `tests/test_gradients.py` de analytiske gradientene med numeriske gradienter (sentrale differanser) på et lite nettverk. Alle relative feil ligger godt under 1e-6:

| Lag     | Vekter (W) | Bias (b) |
| ------- | ---------- | -------- |
| Dense 1 | 3.22e-08   | 6.24e-11 |
| Dense 2 | 1.41e-10   | 1.43e-11 |

## Kjøring

```bash
git clone https://github.com/EirikAarrestad/nn-from-scratch.git
cd nn-from-scratch
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python train.py                    # trener og evaluerer modellen
python -m tests.test_gradients     # kjører gradient-sjekken
```

MNIST lastes ned via scikit-learn første gang og lagres i `data/mnist.npz`, så senere kjøringer går raskt.

## Filer

```
nn-from-scratch/
├── nn/
│   ├── layers.py         # Dense-lag
│   ├── activations.py    # ReLU, Sigmoid, Softmax
│   ├── losses.py         # cross-entropy og one-hot
│   ├── optimizers.py     # SGD
│   └── model.py          # Sequential: fit, predict, evaluate
├── tests/
│   └── test_gradients.py # gradient-sjekk
├── data.py               # laster og lagrer MNIST
├── train.py              # trener og evaluerer modellen
└── requirements.txt
```

## Videre arbeid

- Valideringssett for å følge med på overtilpasning
- Momentum og Adam
- Plott av tapskurven og av feilklassifiserte siffer

## Forfatter

Eirik Aarrestad, bachelorstudent i statistikk og data science ved Universitetet i Bergen.
