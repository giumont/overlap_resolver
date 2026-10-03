# Progetto CERN - Settembre 2026

## 1) Introduzione al Machine Learning, con sviluppo e test di modelli
Link al materiale dell'insegnamento [Metodi di Machine Learning per la Fisica](https://corsi.unige.it/off.f/2025/ins/87930) 

Link a progetti d'esempio esistenti, da studiare ed estendere

**Obiettivo 1.1:** riprodurre ed estendere una selezione di esempi trattati nel corso

## 2) Applicazione all'identificazione di $b$-jet e $\tau$-jet
[Presentazione](https://docs.google.com/presentation/d/1MdQ2Vr2xTQ7ey58RPghjH_iv5nxG66UoTGH4_PSeqvk/edit?usp=sharing) introduttiva su LHC, ATLAS e jet tagging

Per approfondimenti su identificazione dei decadimenti adronici del $\tau$ e del sapore dei jet:
* [Transforming jet flavour tagging at ATLAS](https://arxiv.org/abs/2505.19689)
* [GN3: Multi-task, Multi-modal Transformers for Jet Flavour Tagging in ATLAS](https://cds.cern.ch/record/2953652/files/ATL-PHYS-PUB-2026-001.pdf)
* [Identification of hadronically decaying tau leptons with the ATLAS detector](https://cds.cern.ch/record/2827111/files/ATL-PHYS-PUB-2022-044.pdf)

**Obiettivo 2.1:** comprendere e saper spiegare l'uso di transformer per questa applicazione fisica
**Obiettivo 2.2:** sviluppare un modello che sappia discriminare jet beauty da jet light usando le informazioni sulle tracce che essi contengono

## 3) Applicazione alla ricerca del processo $HH\to bb\tau\tau$
Ultimo [risultato pubblico](https://arxiv.org/abs/2404.12660) della collaborazione ATLAS su questa ricerca

**File di dati:**
* segnale $HH\to bb\tau\tau$ prodotto nel canale gluon-gluon fusion (GGF)
```/eos/user/c/cschiavi/HHBBTT/NTUPLES/VBF-TRIG/GGF-mc23a-bypass-noOR_mode/```
* fondo $t\bar{t}$ completamente adronico
```/eos/user/c/cschiavi/HHBBTT/NTUPLES/VBF-TRIG/TTBAR-mc23a-bypass-noOR_mode```

**Esempio di codice per la manipolazione dei dati:**
https://gitlab.cern.ch/parodi/b-tau_studies

**Obiettivo 3.1:** sul campione di segnale, capire la correlazione fra l'identificazione di $b$-jet e $\tau$-jet, quantificando l'overlap fra le due tipologie di oggetti ricostruiti, confrontando le loro caratteristiche cinematiche e i discriminanti di identificazione ricostruiti dagli algoritmi offline

**Obiettivo 3.2:** sul campione di segnale, quantificare la bontà della classificazione, prima separatamente per $b$-jet e $\tau$-jet e poi provando a combinare "manualmente" i discriminanti di identificazione in caso di overlap

**Obiettivo 3.3:** sul campione di segnale, ottimizzare la definizione dei discriminanti per $b$-jet e $\tau$-jet tramite lo sviluppo di una semplice NN che agisca in caso di overlap

**Obiettivo 3.4:** studiare la separazione fra eventi di segnale e fondo basandosi sul conteggio degli oggetti selezionati tramite i discriminanti per $b$-jet e $\tau$-jet

**Obiettivo 3.5:** ottimizzare la separazione fra eventi di segnale e fondo tramite lo sviluppo di un modello di ML

## 4) Applicazione all'identificazione di coppie di jet boosted - OPZIONALE

Ultimo [risultato pubblico](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PUBNOTES/ATL-PHYS-PUB-2026-013/) della collaborazione ATLAS su questa tecnica di ricostruzione

**Obiettivo 4.1:** studiare le performance in funzione di diversi valori delle flavor fraction che definiscono il discriminante

**Obiettivo 4.2:** effettuare uno scan delle flavour fraction in funzioni delle reiezioni del fondo ad efficienza di segnale fissata 

**Obiettivo 4.3:** studiare GN3X come top tagger e paragonarlo allo stato dell'arte in ATLAS


# Logbook

### Stato di avanzamento
#### Obiettivo 2.2
Bkg rejection su working points a parità di signal efficiency (test set) :
![](https://codimd.web.cern.ch/uploads/upload_4170f5bd2904f7d6d7c472a6216445d0.png)

<!--
| target eff. |  best light-jets (1e6 jets, `MC23G`) | DL1d c-jets (Z')(*) | DL1d light-jets (Z') (*)|
|:-----------:|:-----------------------------:|:----------------:|:--------------------:|
|     60%     |             9.08              |      ~ 4.5       |         ~         |
|     70%     |             6.16              |      ~ 3.0       |         ~         |
|     77%     |             4.62              |      ~ 2.3       |        ~          |
|     85%     |             3.12              |      ~ 1.8       |        ~          |
|     90%     |             2.37              |      ~ 1.5       |        ~           |
|             |                               |                  |                      |
-->
(*) valori estrapolati da Figura 2 paper [Transforming jet flavour tagging at ATLAS](https://arxiv.org/abs/2505.19689)

#### Obiettivo 3.3 e 3.4
Feature con potere discriminante sul segnale $HH\to bb\tau\tau$ come input della DNN (e quali categorie discriminano). In **grassetto** quelle che mantengono lo stesso potere discriminante anche sul fondo $t\bar t$, ~~barrate~~ quelle che lo perdono /presentano distribuzioni evidentemente diverse sul fondo, *in corsivo* quelle dove la poca statistica non permette di trarre conclusioni:
- Variabili singoli oggetti reco:
    - ~~Pseudorapidità $\eta_\tau, \eta_{jet}$~~: *(FF) VS (TT, TF, FT)*
    - ~~$MET_{\parallel,\tau}, m_{T,\tau}$ (ricavata)~~: *(_T) VS (_F)*
    - **numero muoni soft** (`recojet_n_muons`): *(T_) VS (F_)*
    - **massa del jet** (`recojet_mass`): *(T_) VS (F_)*
    - *decadimento del $\tau$* (`tau_decayMode`, `tau_nProng`): *(TF) VS (FT)*
    - *$p_T$ di $\tau$ e $p_T$ del jet reco*: *(FF) VS (TT, TF, FT)*
    
- Variabili cinematiche relative oggetti reco:
    - *Angolari $\Delta \eta, \Delta \phi$*: *(FT) VS (TT, TF, FF)*
    - *Parametro $\Delta R$*: *(FT) VS (TT, TF, FF)*
    - ~~Rapporto $p_{T, jet} / p_{T, \tau}$~~: *(FT) VS (TT, TF, FF)*

- Taggers: 
    - Score GN2 (quantili): *(T_) VS (F_)*
    - Scores $\tau$-taggers GNNTau e/o RNNJetScore (continuo): *(FT) VS (TT, FF, TF)*
    - [*Nota*: Score GN2 + $\tau$-\tagger (correlazione): *(TT) VS (FT, FF, TF)*]

    [***come tagger tau, si potrebbe prendere solo GNTau: performance migliori di RNNJetScore (es: su AUC 0.9780 VS 0.9650), e il comportamento al variare della categoria di verità sugli score sembra essere lo stesso; si potrebbe fare analisi correlazione per esserne sicuri***]
    [***UPDATE: alla fine ho usato tutti i tagger come input della rete***]

Valutate ma non discriminanti sul segnale (non verranno usate per la rete):
- $\phi_\tau, \phi_{jet}$
- `tau_charge`
- numero vertici primari ricostruiti: `nPrimaryVertices`

**Obiettivo 3.3 - Best model globale (non su classe TT)**
Utilizzando Focal Loss con scelta di parametri ($p_\alpha = 0.5, \gamma = 1$) [link alla run](https://github.com/giumont/overlap_resolver/tree/main/outputs/DNN/HH_bbtt_dataset/SWEEP_FOCAL_PARAMS_2layers_32_16_dropout0.1_LayerNorm_batch2048_lr0.001_wd0.0001_best_on_ap_macro_criterion_CategoricalFocalLoss()_weight_loss_True_sampling_strategy_none/focal_a0.5_g1) si ottengono i seguenti risultati: 
    ![](https://codimd.web.cern.ch/uploads/upload_c383b29df4aa0a4dbbce6e02eb3e4e27.png)
    ![](https://codimd.web.cern.ch/uploads/upload_2642383bc77f307b6a84a7176418efcc.png)
    ![](https://codimd.web.cern.ch/uploads/upload_c8810d4d97337a24a0e171366005c200.png)

### Materiale e codice prodotto
- [**Repository Github (privata) obiettivo 2.2**](https://github.com/giumont/flavour_tagging/tree/main/notebooks)
- [**PPT presentazione 15/09 progressi obiettivi 2.2**](https://docs.google.com/presentation/d/163YeiBd-aiiwa2YYRg1pccQ4OY1vI3mqLkX9deiRN4g/edit?usp=sharing)

- [**Repository Github (privata) obiettivo 3.3**](https://github.com/giumont/overlap_resolver)

<!-- Struttura repo:
```
flavour_tagging/
├── notebooks/
│   ├── H5_files_reading.ipynb              # [DRAFT] Esplorazione della struttura e del contenuto dei file HDF5.
│   ├── Obj2_single_tracks.ipynb            # Studio preliminare dell'obiettivo 2 con dataset di tracce singole.
│   ├── Obj2_single_tracks 1e5.ipynb        # [DRAFT] Training/analisi su un campione di 10^5 jet.
│   ├── Obj2_single_tracks_1e6.ipynb        # [DRAFT] Training/analisi su un campione di 10^6 jet.
│   └── preprocessing/
│       └── Preprocessing_h5_MC23G.ipynb    # Preprocessing del dataset MC23G e produzione dei file HDF5.
│
├── outputs/
│   └── DNN_jets1e5_SALT_DATASET/  # [DRAFT]
│
├── README.md                               # Descrizione del progetto e istruzioni d'uso.
│
└── src/
    ├── checkpoint_io.py                    # Salvataggio e caricamento dei checkpoint del modello.
    ├── data.py                             # Caricamento e preparazione dei dataset.
    ├── graphics.py                         # Funzioni per la generazione dei grafici.
    ├── __init__.py                         # Inizializzazione del pacchetto Python.
    ├── merge_datasets.py                   # Unione di dataset provenienti da file differenti.
    ├── models.py                           # Definizione delle architetture di rete neurale.
    ├── training.py                         # Pipeline di addestramento e validazione del modello.
    └── utils.py                            # Funzioni di utilità.```
```
-->

### Problemi 
#### Prioritari
- [ ] Poca statistica su $HH\to bb\tau\tau$ relativi alle categorie true+true e fake+fake

- [ ] Accesso a GPU (Colab si esaurisce)
- [x] Sistematico overfitting validation set Obiettivo 2.2

#### Secondari
- [ ] Errori nell'esecuzione notebooks didattici su Colab

- [ ] Chiarire significato di `tau_tauTruthJetLabel` e valori associati, per `.root` di $HH\to bb\tau\tau$
- [ ] Validare ipotesi: "la mask `parentHiggsParentsMask` è posta uguale a zero per tutti i $\tau$ che sono ricostruiti come jet (`reco_jets_`) ma non come singole particelle $\tau$ (`tau_`)"

#### Risolti
- [x] ~~Fallito accesso ai file di dati punto 3 (SSH)~~

### Prossimi passi

**Obiettivo 2.2**
- Preprocessing MC23G: applicare cut su significanze basate su percentili / statistiche del trainset (*cut sul valore numerico della grandezza fisica, non sulla soglia di percentile*) e reintrodurli come features nel training
- Studiare score distribution per c jets nel classificatore binario b/light implementato
- Procurare dataset MC23G eventi $t\bar t$ e ripetere training con best model: 
    - confrontare metriche con training su $Z'$ e con risultati ATLAS
    - studiare distribuzione $p_T$ del jet prima e dopo aver applicato il tagger (e volendo ripetere su $Z'$, ma più interessante qui)
- Approfondire concetto "running statistics" con BatchNorm1D e come/se questo potrebbe spiegare run patoogiche risolte o migliorate con LayerNorm

**Obiettivo 3.1**
- Studiare la composizione del caso ($\tau$ vero, $b$-jet fake -> label `TrustTau` per la rete) in overlapping: per questi $b$-jet fake, la sottocategoria con `HadronConeExclTruthLabelID==15` corrisponde con tutta la categoria, oppure c'è una parte rilevante che non ha né flavour $b$ né "flavour" $\tau$?
*Motivazioni*:
    - se esiste una sottocategoria `HadronConeExclTruthLabelID!=15` (e `HadronConeExclTruthLabelID!=5`) consistente, la DNN potrebbe avere più difficolta ad aggregare in una unica label `TrustTau` questo caso e quello in cui `HadronConeExclTruthLabelID==15` (ovvero quando c'è overlap genuino)

- Studiare la composizione del caso ($\tau$ vero, $b$-jet vero -> label `Altro (entrambi veri)`) in overlapping: per questi $\tau$ veri, la sottocategoria con `tau_tauTruthJetLabel==5` corrisponde a tutta la categoria, oppure c'è una parte rilevante che ha un valore diverso per questo parametro?
*Motivazioni*:
    - se grosso modo `tau_tauTruthJetLabel==5` per tutti i $\tau$ di questa categoria, sarebbe un forte indizio a favore dell'interpretazione di questa label come "flavour di verità del jet più vicino al $\tau$ reco"

<!--
2. Altre analisi incrociate suggerite

Priorità alta — le più direttamente utili:

Confronto cinematico diretto tra il jet fake e il τ vero-matched-truth, nella categoria "jet fake, τ vero": prendi tau_truth_pt_vis/eta_vis/phi_vis (il quadrivettore truth-visibile già associato al tuo reco-τ) e confrontalo con il quadrivettore reco del jet fake nella stessa coppia (pt/eta/phi del jet, non la verità del jet). Se sono quasi identici (entro la risoluzione tipica di calorimetro/tracciatore), è la prova quasi diretta che il jet è semplicemente una ricostruzione alternativa dello stesso decadimento del τ — un test molto più stringente della semplice vicinanza ΔR tra due reco generici. Motivazione: questo è l'unico modo per distinguere "stesso oggetto ricostruito due volte" da "due oggetti reco che per caso stanno vicini", perché confronta la ricostruzione con la verità invece che due ricostruzioni indipendenti tra loro.
Matching di posizione tra il jet label==15 e la collezione completa truthtau_eta_vis/phi_vis, confrontato con il match truth del τ specifico nella coppia: è la procedura completa che rispondere in modo definitivo alla tua domanda 3 — vedi sezione dedicata sotto. Utilità massima, perché fornisce l'unico criterio di verità realmente non ambiguo che hai a disposizione nel dataset per la domanda "stesso oggetto sì/no".
parentHiggsParentsMask del jet vs del τ, nella sotto-popolazione già isolata al punto (a)/(b): hai già notato un comportamento poco chiaro (l'80% dei jet label==15 ha mask==0, nonostante i τ veri siano associati ad un H praticamente sempre). Ripetere questo confronto ristretto alla sola regione di overlap geometrico (invece che sull'intera popolazione inclusiva come nelle tue note precedenti) potrebbe chiarire il mistero: se scopri che gli eventi label==15/mask==0 sono concentrati fuori dalla regione di overlap (cioè non sono affatto vicini a un vero τ), la spiegazione più probabile è che siano jet tau-labeled per ragioni indipendenti dagli H1/H2 di questo canale (es. τ da altri processi/pileup non associati all'Higgs), mentre se sono concentrati dentro l'overlap, rafforza l'ipotesi di un problema di propagazione della mask specifico per questa sottocategoria. Motivazione: riusa un'osservazione già aperta nelle tue note, restringendo il campione a un sottoinsieme più controllato e quindi più interpretabile.

Priorità media:

Molteplicità locale come controllo: nella categoria "jet fake + τ vero" con label==15 (duplicazione sospetta), verifica se il numero di altri jet/τ nello stesso evento è sistematicamente più basso rispetto alla categoria "entrambi veri" (eventi genuinamente più affollati, dove è più plausibile che due oggetti distinti finiscano vicini per combinatoria). Utilità minore rispetto ai punti 1-2, ma economico da calcolare e utile come cross-check indipendente.

Come ottenere un criterio davvero certo, dal dataset che hai:

L'unico modo per sapere con certezza se il jet e il τ della coppia sono ricostruzioni dello stesso oggetto fisico è smettere di confrontare reco-vs-reco (ΔR tra le due ricostruzioni) e confrontare invece quale oggetto di verità sta dietro a ciascuno dei due, separatamente:

Per il τ reco: hai già l'informazione pronta — tau_truth_eta_vis, tau_truth_phi_vis, tau_truth_pt_vis, tau_truth_m_vis sono il quadrivettore visibile del τ di verità specifico già associato (matched) a quel particolare τ reco.
Per il jet reco: non hai un branch equivalente diretto (bJetTruthDR/Pt sono specifici per il matching a adroni-b, non a τ), ma puoi ricostruirlo: se il jet ha HadronConeExclTruthLabelID==15, la label stessa è stata assegnata da ATLAS tramite un matching a cono (tipicamente ΔR<0.3) tra l'asse del jet e un τ adronico di verità. Puoi riprodurre lo stesso matching confrontando (eta, phi) del jet con l'intera collezione truthtau_eta_vis/truthtau_phi_vis, entro lo stesso raggio di cono, per identificare quale τ di verità specifico ha originato quella label.
Confronto finale: se il τ di verità trovato al punto 2 (quello che ha "acceso" la label 15 del jet) e il τ di verità trovato al punto 1 (quello associato al τ reco della coppia) sono lo stesso oggetto (stesso quadrivettore, entro tolleranza numerica trascurabile — idealmente stesso indice nella collezione truthtau_) → hai la certezza che jet e τ reco sono due ricostruzioni indipendenti dello stesso decadimento fisico. Se invece puntano a due τ di verità diversi (o uno dei due non ha alcun match) → sono, con certezza, due oggetti fisicamente distinti, indipendentemente da quanto piccolo sia il loro ΔR reco-reco.
-->



### Bibliografia utilizzata
- [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)


- **Procedura di standardizzazione dei dataset e data-leakage**:
    - [scikit-learn User Guide: Common Pitfalls and Recommended Practices](https://scikit-learn.org/stable/common_pitfalls.html?highlight=linearregression&utm_source=chatgpt.com)
    - [scikit-learn Documentation - StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html?utm_source=chatgpt.com)
    - [Forum StackOverflow - Data Leakage Concerns](https://stats.stackexchange.com/questions/522592/data-leakage-concerns)

- **Batch normalization VS Layer normalization**: [BOZZA]
    - [Pagina reddit](https://www.reddit.com/r/deeplearning/comments/1ozps3j/when_should_batchnorm_be_used_and_when_should/)

- **Gestione class-imbalances**:
    - [Class Imbalance Techniques for High Energy Physics](https://arxiv.org/abs/1905.00339?utm_source=chatgpt.com)
    - [Repo][Class-Imbalance-in-WW-Polarization](https://github.com/christopher-w-murphy/Class-Imbalance-in-WW-Polarization)
    
    - **Evaluation metrics**
        - [DA VEDERE][Blog - Tour of Evaluation Metrics for Imbalanced Classification](https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/)

        - [DA VEDERE] [Medium - Use weighted loss function to solve imbalanced data classification problems](https://medium.com/@zergtant/use-weighted-loss-function-to-solve-imbalanced-data-classification-problems-749237f38b75)
        
        - [DA VEDERE] [Blog - Focal Loss : A better alternative for Cross-Entropy](https://towardsdatascience.com/focal-loss-a-better-alternative-for-cross-entropy-1d073d92d075/)
        - [Blog - How F1 score is good with unbalanced datasets](https://datascience.stackexchange.com/questions/105089/how-f1-score-is-good-with-unbalanced-dataset)
        - [Reddit - How is F1-score a better metric for unbalanced data sets ?](https://www.reddit.com/r/MachineLearning/comments/ushyhd/r_how_is_f1score_a_better_metric_for_unbalanced/)
        - [Blog - How can the F1-score help with dealing with class imbalance?](https://sebastianraschka.com/faq/docs/computing-the-f1-score.html)
    
    - **Data sampling**
        - [DA VEDERE][Blog - Tour of Data Sampling Methods for Imbalanced Classification](https://machinelearningmastery.com/data-sampling-methods-for-imbalanced-classification/)
        - [DA VEDERE][Medium - Demystifying PyTorch’s WeightedRandomSampler by example](https://medium.com/data-science/demystifying-pytorchs-weightedrandomsampler-by-example-a68aceccb452)

- **Letteratura su b-tagging**:
    - [DA VEDERE] [GN3: Multi-task, Multi-modal Transformers for Jet Flavour Tagging in ATLAS](https://cds.cern.ch/record/2953652/files/ATL-PHYS-PUB-2026-001.pdf) (2026): **schema e dettagli architettura GN3** + **feature usate in GN3 e confronto con quelle di GN2** + plot $R_{bkg}(\epsilon_b)$ per GN3 vs GN2 in regime $t\bar t$ e $Z'$
    - [Transforming jet flavour tagging at ATLAS](https://arxiv.org/abs/2505.19689) (2026): **schema e dettagli architettura GN2** + confronto architettura con DL1 + **plot $R_{bkg}(\epsilon_b)$ per GN2 vs DL1 in regime $t\bar t$ e $Z'$**
    - [Machine Learning in High Energy Physics: A review of heavy-flavor jet tagging at the LHC](https://link.springer.com/article/10.1140/epjs/s11734-024-01234-y) (2024): sezione introduttiva al ML e alle principali architetture (DNNs, CNNs, RNNs, Deep Sets, GNNs) + applicazioni a flavour-tagging (fino a GN2)
    - [Flavour tagging with graph neural networks with the ATLAS detector](https://cds.cern.ch/record/2811135?ln=it) (2023): presentazione di GL1 e **confronto architettura con DL1r** + **tabella comparativa iperparametri tra GL1 e GN2** 
    - [ATLAS flavour-tagging algorithms for the LHC Run 2 pp collision dataset](https://cds.cern.ch/record/2842028/files/document.pdf) (2023): dettagli su low-level b-taggers e loro uso per high-level flavour taggers (serie DL1, DL1r), con **dettagli (tabelle iperparametri) questa architettura**
    - [Deep Learning in Flavour Tagging at the ATLAS experiment](https://cds.cern.ch/record/2274065) (2017): presentazione di DL1, info generali sull'architettura; da approfondire lo **scan sugli iperparametri**:
        > Building upon the methods described previously for stabilising and improving the learning
        > process, and preventing overtraining, a **systematic grid search** has been performed on the number
        > of hidden layers, the number of nodes in the hidden layers, the sequencing of Maxout and Dense
        > layers as well as the learning rate. 

 - **Letteratura su tagging decadimenti adronici $\tau$**:
     - [DA VEDERE] [Reconstruction, Identification, and Calibration of hadronically decaying tau leptons with the ATLAS detector for the LHC Run 3 and reprocessed Run 2 data](https://cds.cern.ch/record/2827111/files/ATL-PHYS-PUB-2022-044.pdf)

- **Altro**:
    - [Lecture slides - Electroweak physics at LEP](https://www0.mi.infn.it/~fanti/Particelle3/05-PhysicsAtLEP.pdf?utm_source=chatgpt.com): slide "b-tagging: lifetime and impact parameter" (n.59)



## Lun 2026-08-31

### Attività
**Obiettivo 1.1:**
- Studio materiale del corso "Metodi di Machine Learning per la Fisica": 
    * ripasso/studio cap.1 ("Introduction")
    * ripasso/studio cap.4 ("Introduction to Neural Networks") 
    * inizio studio cap. 5.5 ("Graph Neural Networks")

**Obiettivo 3.1:**
- Clone della repo [Jet-Tau matching studies](https://gitlab.cern.ch/parodi/b-tau_studies) e visualizzazione preliminare del codice

**Altro - Burocrazia:**
- Ottenuta access card CERN
- Richiesta chiave ufficio 40/2-C01 tramite portale CERN

### Risultati
--

---

## Mar 2026-09-01

### Attività
**Obiettivo 1.1:**
- Studio materiale del corso "Metodi di Machine Learning per la Fisica": 
    * studio cap. 5.5 ("Graph Neural Networks")
- Inizio studio esempio architettura GNN + CNN per applicazione fisica [GNN_tutorial_2025](https://github.com/ML4PhysicsTeachingGenova/course_2025/blob/main/GNN_tutorial_2025.ipynb)

**Obiettivo 2.1:**
- Introduzione Prof. Coccaro: concetti di *determinant, efficiency, rejection, ROC* e applicazioni ML (Transformer) per jet flavour tagging in ATLAS

**Altro - Risoluzione problemi:**
- Fallito accesso a SSH: aperto ticket CERN ServiceDesk

### Risultati
--

---

## Mer 2026-09-02

### Attività


**Obiettivo 2.1:**
- Studio paper GN2 [Transforming jet flavour tagging at ATLAS](https://arxiv.org/abs/2505.19689): 
    * comprensione di base architettura modello
    * parametrizzazione tracce (*pT, d0, z0*)

**Obiettivo 2.2:**

- Ottenimento dataset (nel seguito ```salt-tutorial```) da directory
```/eos/atlas/atlascerngroupdisk/perf-flavtag/training/salt-tutorial ./ ```
- Analisi preliminare struttura dati (formato HDF5) e potere discriminante osservabili tra jet beauty e light:
    - Istogramma occorrenze per variabili cinematiche globali sui jets (key ```["jets"]``` nel file HDF5```salt-tutorial```)
    - Istogramma occorrenze per parametri di impatto singole tracce (key ```["consts"]``` nel file HDF5 ```salt-tutorial```)
- Test e verifiche di correttezza sull'implementazione delle funzioni di lettura dei file HDF5:
    - Test su labels associate a flavour light, charm e beauty (parametro ```["jets"]["flavour_label"]``` nei file HDF5 ```salt-tutorial```) utilizzando la distanza di decadimento trasversa associata alle diverse label (parametro ```["hadrons"]["Lxy"]``` nei file HDF5 ```salt-tutorial```)

**Obiettivo 3.1:**

- Ottenimento dati da directory
```/eos/user/c/cschiavi/HHBBTT/NTUPLES/VBF-TRIG/GGF-mc23a-bypass-noOR_mode/```

**Altro - Burocrazia:**
- Ritirata chiave ufficio 40/2-C01

**Altro - Risoluzione problemi:**
- Fallito accesso a SSH: RISOLTO

### Risultati

**Obiettivo 2.2:**
- **PRELIMINARE - Potere discriminante delle feature**:
I parametri di impatto delle tracce (es: *$d_0$, $z_0$*) mostrano (qualitativamente) più potere discriminante tra b e l rispetto alle variabili cinematiche globali sui jets (es: *$p_T$, $\eta$*) 
    - Rilevanza: selezione delle feature nell'addestramento di NN per b-tagging
    
- **OSSERVAZIONE - Potere discriminante dei parametri di impatto delle tracce su dataset ```salt-tutorial```**
Per lo specifico dataset considerato (```salt-tutorial```), il plotting dei parametri di impatto delle tracce mostrano meno potere discriminante di quanto atteso tra beauty e light. In particolare:

     -  `d0` (*parametro di impatto trasverso*) — distribuzioni quasi sovrapposte tra b e light
         Ci si aspetterebbe distribuzione più piccata per light rispetto a b: questa distinzione è presente, ma la separazione è minima (asse y in scala logaritmica)
    ![](https://codimd.web.cern.ch/uploads/upload_57da242877a1445f3f9766bc6ddaa3d0.png) 

    - `signed_2d_ip` (*parametro di impatto 2D firmato*) - asimmetria positiva anche per light
        Entrambi i flavour presentano la coda destra che dovrebbe essere tipica del b (decadimento displaced dell'adrone), mentre per light ci si aspetterebbe distribuzione simmetrica attorno allo 0
    ![](https://codimd.web.cern.ch/uploads/upload_edb056de7ab5ffedc6b7fa2ac7fa4716.png)

        - `signed_3d_ip` (*parametro di impatto 3D firmato*) - struttura bimodale anomala
        Per entrambi i flavour si osservano due picchi simmetrici rispetto allo 0 al posto di una distribuzione centrata attorno a questo valore
     ![\small](https://codimd.web.cern.ch/uploads/upload_a516645eb490e5e6acccd45f683fabeb.png)


        Il test sulla distanza di decadimento trasversa conferma che l'assegnazione delle labels nel dataset ai corrispondenti flavour è corretta, escludendo questo tipo di bug; la distribuzione di questa osservabile per come i flavour light, charm e beauty sono labellati nel codice è:
        - light: $0 \lesssim L_{xy}$
        - charm: distrbuzione con picco $\sim 0$ e coda destra 
        - beauty: distribuzione con picco $\sim 0$ e coda destra più pesante dei charm
    [AGGIUNGERE IMMAGINI]




---

## Gio 2026-09-03

### Attività

**Obiettivo 2.2:**

- Addestramento DNN per classificazione binaria b/light (```salt-tutorial```): 
    - creazione notebook colab
    - implementazione scheletro del programma: 
        - funzioni di lettura file 
        - preprocessing features e labels
        - definizione modello (MLP)
        - training loop

**Altro - Attività formative:**
- Visita ATLAS 
- Partecipazione lezione: tema detector design per FCC 


### Risultati
--

---

## Ven 2026-09-04

### Attività

**Obiettivo 2.2:**

- Addestramento DNN per classificazione binaria b/light (```salt-tutorial```): 
    - tentativi di ottimizzazione parametri per:
        - trainset 100.000 jets
        - trainset ridotto 5.000 jets (sanity check)
        - trainset e validation set estratti da stesso file (sanity check)
    - produzione output grafici: 
        - loss (BCE) su train set e validation set
        - confusion matrix

### Risultati

**Obiettivo 2.2:**
- **PARZIALE - Loss su trainset e overfitting su validation set**
    * Le configurazioni DNN, da una prima ottimizzazione sui diversi trainset considerati, raggiungono prestazioni accettabili in termini di loss sul training set, ma mostrano sistematicamente un marcato overfitting sul validation set. 
    In particolare, la validation loss inizia ad aumentare già dopo un numero ridotto di epoche (~1–5), nonostante la loss sul training set continui a diminuire, indicando una capacità di generalizzazione peggiore rispetto al caso random.
    * I sanity check effettuati, consistenti nell'addestramento su dataset di dimensioni ridotte e nell'inferenza sullo stesso training set, confermano che il modello è effettivamente in grado di apprendere.
    * Si esclude inoltre che il comportamento anomalo osservato per la validation loss sia dovuto a un'incompatibilità tra i dataset di training e validation ottenuti dai file H5 forniti. Infatti, estraendo training set e validation set a partire dallo stesso file H5 di origine, si osserva lo stesso comportamento per entrambe le loss. 


---

## Sab 2026-09-05

---

## Dom 2026-09-06

### Attività 
- Creazione repo privata su github con codice
- Inizio riorganizzazione codice


---

## Lun 2026-09-07

### Attività

**Obiettivo 2.2:**
- Riorganizzazione del codice in file .py + notebooks 
- Riscrittura + ottimizzazione funzione di lettura file HDF5:
    - Implementazione di un sistema di checkpoint per il salvataggio progressivo degli array NumPy di feature e label estratti dai file HDF5 tramite la libreria ```h5py```. Questo riduce il rischio di perdita dei dati già elaborati in caso di timeout di Colab o di crash durante l'esecuzione.
- Implementazione funzioni grafiche per la valutazione del modello:
    * andamento della Binary Cross-Entropy (BCE) loss su training set e validation set;
    * confusion matrix;
    * curva ROC e relativo AUC;
    * rejection in funzione dell'efficienza;
    * distribuzione degli score del modello per segnale e fondo;
    * efficienza in funzione del cut sullo score
- Addestramento DNN per classificazione binaria b/light (```salt-tutorial```): 
    - tentativi di ottimizzazione parametri per:
        - trainset 100.000 jets 
    - produzione grafici con funzioni implementate (valutazione su testset)
        

### Risultati
**Obiettivo 2.2:**
- **OUTPUT - Addestramento della DNN per trainset di 100.000 jets (dataset ```salt-tutorial```)**:
[DA COMPLETARE]

- **OSSERVAZIONE - Effetto della normalizzazione nei layer della DNN su dataset ```salt-tutorial``` (pt.1)**
Sostituire la normalizzazione per batch (```BatchNorm1D()```) con la normalizzazione su singolo dato (```LayerNorm()```) nella DNN ha eliminato il problema dell'overfitting immediato sul validation set, ottenendo a parità di altri parametri una loss decrescente anche per il validation set. Questa differenza di comportamento si è ottenuta consistentemente al variare di altre caratteristiche del modello:
    - **Numero di hidden layers**: all'aumentare del numero di layers, la differenza tra loss sul trainset (decrescente) e sul validation set (crescente) a parità di epoche è più spiccata.
         - Valori provati: 
         ``` (64, 32) o (128, 64, 32) o (256, 64, 32) ```

        All'incirca, dopo 50 epoche:

        - per (128, 64, 32): su trainset loss~0.4, su valset~0.9
        - per (256, 128, 64, 32): su trainset loss ~0.1, su valset oltre 2  
    - **Regolarizzazione L2**: aumentare la regolarizzazione, tramite ```weight_decay``` in 
  ```  optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)```
  non modifica l'andamento monotono crescente della loss sul valset, lo rallenta solo.
        - Valori provati: ```[1e-4, 1e-5, 1e-6]```
    - **Dropout**: come per regolarizzazione L2; implementata tramite 
    ``` nn.Dropout(dropout) ```
    in ogni layer della DNN.
        - Valori provati: ```[0, 0.1, 0.2, 0.3]```
    - **Learning rate**: aumentare il LR tramite 
    ``` optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) ```
    aumenta la diminuzione della loss sul trainset a parità di epoche, ma non modifica sostanzialmente l'outcome 
        - Valori provati: ``` [1e-3, 3e-3, 1e-2]```
  
  
Questo risultato è interessante, e potrebbe essere dovuto a dimensioni effettive del dataset troppo basse (che influenzano il comportamento di ```BatchNorm1D()```, ma non quello di ```LayerNorm()```). 
Sarà prioritario osservare l'effetto delle due diverse normalizzazioni per datasets di dimensioni maggiori (~1.000.000 jets).

---
## Mar 2026-09-08

### Attività

**Obiettivo 2.2:**
- Riscrittura + ottimizzazione funzione di lettura file HDF5:
    - Miglioramento del sistema di checkpoint per il salvataggio progressivo degli array NumPy di feature e label estratti dai file HDF5 tramite la libreria ```h5py```: i checkpoint sono file .npz separati, i metadati sull'avanzamento del processo di salvataggio sono salvati in un altro file ("manifest") più leggero; questo evita l'uso esponeziale della RAM all'aumentare del numero di checkpoint salvati.

- Addestramento DNN per classificazione binaria b/light (```salt-tutorial```): 
    - tentativi di ottimizzazione parametri per:
        - trainset 1.000.000 jets
    - produzione grafici con funzioni implementate (valutazione su testset)
    [IN SOSPESO]
    
 - Ottenimento dataset relativo a campagna MC23g, "Extended $Z'$ with Gluons & Hadronic Taus" (nel seguito ```MC23G```); H5 ntuple:
```group.perf-flavtag.802818.e8514_s4618_r17610_p7349.tdd.GN3_dev.25_2_101.26-08-20_Central_Production_output.h5 ```
(si veda [ATLAS Flavour Tagging Documentation](https://ftag.docs.cern.ch/samples/h5_sample_list/#centrally-preprocessed-h5-samples-for-general-trainings))
    - Trasferimento di un sottoinsieme casuale di file dati HDF5 presenti nel database (per un totale di 16.769.606 jets) su google drive

### Risultati
**Obiettivo 2.2:**
- **OUTPUT - Addestramento della DNN per trainset di 1.000.000 jets (dataset ```salt-tutorial```)**:
[DA COMPLETARE]

- **OSSERVAZIONE - Potere discriminante dei parametri di impatto delle tracce su dataset ```MC23G```**
    - `d0` (*parametro di impatto trasverso*)
    La distribuzione per i light è più piccata attorno allo zero, quella per i b ha code più pesanti (come atteso)
![\small](https://codimd.web.cern.ch/uploads/upload_435abff6d06e4958695c13a80b54628e.png)
    - `lifetimeSignedD0Significance` (*significanza del parametro di impatto 2D firmato*)
    La distribuzione per il flavour b presenta una coda positiva dovuta al tempo di decadimento $\sim ps$ del beauty, non presente per i light
    ![](https://codimd.web.cern.ch/uploads/upload_96723592620c9e939c156b928945d846.png)
        Riducendo il range per comparare con un plot ufficiale ATLAS (Figura 1 [ATLAS flavour-tagging algorithms for the LHC Run 2 pp collision dataset](https://cds.cern.ch/record/2842028/files/document.pdf) c'è corrispondenza (*si veda scheda 15/09 per approfondimento sul perché sono comparabili*):
        ![](https://codimd.web.cern.ch/uploads/upload_fdd1e70ea9dc6d3abd639f77f7dd10f5.png)
        ![](https://codimd.web.cern.ch/uploads/upload_5c94232a0cc57452d3f1f3370e0135e8.png)

    - `lifetimeSignedZ0SinThetaSignificance` (*significanza del parametro di impatto 3D firmato*)
    Le distribuzioni sono centrate intorno allo zero, come atteso
    ![](https://codimd.web.cern.ch/uploads/upload_77a8c0ab42c386188a4bc9da3e78b810.png)


- **OSSERVAZIONE - Effetto della normalizzazione nei layer della DNN su dataset ```salt-tutorial``` (pt.2)**
Aumentare le dimensioni dei dataset da 100.000 a 1.000.000 jets NON ha eliminato il problema dell'overfitting immediato su validation set quando si utilizza ```BatchNorm1D()```. 
Si procederà a training dello stesso modello su diverso dataset (```MC23G``` al posto di ```salt-dataset```) per confronto: si ipotizza che l'anomalia sul training per ```salt-dataset``` possa essere dovuto ~~ad una etichettatura errata dei jets nel file HDF5 originario~~, ~~coerentemente con la precedente osservazione sullo scarso potere discriminante dei parametri di impatto tra flavour.~~ (UPDATE: l'unica cosa ipotizzabile è che il potere discriminante fosse troppo basso; l'ettichettatura può difficilmente essere messa in discussione). Il dataset ```MC23G``` presenta maggiore potere discriminante tra questi parametri, quindi verrà usato per testare questa ipotesi.

---

## Mer 2026-09-09

### Attività

**Obiettivo 2.2**
- Funzione di lettura file HDF5:
    - Generalizzazione della funzione a file HDF5 con struttura interna generica
- Preprocessing del dataset `MC23G`:
    - Estrazione delle feature di interesse per i jet light e beauty dai file HDF5 di input:
    ```python
    [
    "d0",                                    # parametro di impatto trasverso
    "z0SinTheta",                            # parametro di impatto longitudinale (z0 *  sin(theta))
    "dphi",                                  # phi relativo traccia-jet
    "deta",                                  # eta relativo traccia-jet
    "pt",                                    # pt della traccia
    "lifetimeSignedD0Significance",          # significativita' del signed 2D IP
    "lifetimeSignedZ0SinThetaSignificance",  # significativita' del signed 3D IP (z0)
    "qOverP",                                # carica / momento
    ]
    ```
    a livello di singola traccia (key `["tracks_ghost"]`), con un numero fisso di tracce per     jet (prime 20). Etichettatura dei jet secondo `HadronConeExclTruthLabelID` (light vs beauty).
    - Estrazione eseguita con checkpointing (chunk periodici + "manifest" di avanzamento), per permettere il recovery in caso di crash o riavvio del runtime senza perdere il lavoro già svolto.
    - Merge dei checkpoint di ciascun file `.h5` in un unico file per file, con verifica di coerenza dei nomi delle feature tra i vari file.
    - **Bilanciamento** delle classi light/beauty *file per file* (sottocampionamento della classe maggioritaria fino a pareggiare la minoritaria, con shuffle), producendo un file bilanciato per ciascun input.
    - Assegnazione a train/val/test a livello di file interi, distribuendo i file bilanciati tra i tre split rispettando le frazioni target (70/15/15), assegnando ogni file allo split che più si discosta dal proprio obiettivo.
    - Merging finale dei file assegnati a ciascuno split tramite concatenazione su disco via memmap, evitando di tenere l'intero array in RAM; salvataggio intermedio su disco locale e successiva copia su Drive in formato `.npy` non compresso.
    - **Standardizzazione delle feature**: media e deviazione standard calcolate sul solo trainset con un algoritmo streaming a blocchi (ottimizzato per RAM), poi applicate (sempre a blocchi) a train, validation e test set.
    - Sanity check finali su tutti gli split: coerenza delle shape, assenza di NaN/Inf, verifica del bilanciamento delle classi e verifica per-feature della corretta standardizzazione.

**Altro - Burocrazia**
- Ottenuto account presso INFN Genova (per uso risorse di calcolo)


### Risultati
- **OSSERVAZIONE - Distribuzione delle feature nel dataset ```MC23G``` (pt.1)**:
I sanity check finali (numero features: 20 tracce per jets * 8 parametri per traccia = 160) mostrano valori fuori scala per media e std massime su validation e testset dopo la rinormalizzazione con i pesi associati al trainset (media=0, varianza=1). 
    ```text
    === CHECK: TRAIN ===
    Shape X: (9410238, 160) | Shape y: (9410238,)
    Bilanciamento (mean y): 0.5000
    NaN totali: 0 | Inf totali: 0
    mean per-feature: min=-0.0000, max=0.0000
    std  per-feature: min=1.0000, max=1.0000

    === CHECK: VAL ===
    Shape X: (3139442, 160) | Shape y: (3139442,)
    Bilanciamento (mean y): 0.5000
    NaN totali: 0 | Inf totali: 0
    mean per-feature: min=-0.8508, max=5814.7767
    std  per-feature: min=0.0001, max=1311777.7762

    === CHECK: TEST ===
    Shape X: (1871028, 160) | Shape y: (1871028,)
    Bilanciamento (mean y): 0.5000
    NaN totali: 0 | Inf totali: 0
    mean per-feature: min=-0.8452, max=6409.4349
    std  per-feature: min=0.0001, max=845828.2816
    ```
    Da verificare se l'origine di questo comportamento è dovuto ad un bug nel codice per la normalizzazione, ad uno splitting non equilibrato dei jets tra trainset, valset e testset a partire dai file HDF5 grezzi o a singole feature molto rumorose / con valori non validi nei dataset.

     **NOTA**:  Una prova di training con DNN con questo dataset completo è risultata in **loss immediatamente crescente sul validation set, sia per ```LayerNorm``` che per ```BatchNorm1D``` come layer normalization**. Tuttavia, questo risultato non viene considerato informativo ai fini della valutazione delle prestazioni del modello, poiché le anomalie osservate nella distribuzione delle feature tra train, validation e test set non consentono di escludere che il comportamento della rete sia dovuto a problemi nella fase di preprocessing e standardizzazione.
    Il risultato viene pertanto considerato esclusivamente come un'ulteriore evidenza della necessità di individuare e risolvere le anomalie nella distribuzione delle feature prima di procedere alla valutazione delle performance della DNN.

---
## Gio 2026-09-10

### Attività

**Obiettivo 2.2:**
- Lettura documentazione e forum relativi al processo di standardizzazione dei dataset: buone pratiche e problemi di data-leakage
- Preprocessing del dataset `MC23G`:
    - Ricerca, all'interno del file HDF5, di flag di invalidità associati alle singole tracce.
    - Ricerca di feature rumorose o contenenti valori invalidi, mediante l'analisi delle proprietà statistiche (media e deviazione standard) delle singole feature per traccia, calcolate dopo la standardizzazione sulla base del trainset.
    - Rimozione delle feature confondenti e non discriminanti individuate durante il processo di training (3 parametri): nuovi dataset con (8-3) parametri per traccia * 20 tracce per jet = 100 feature.
- Addestramento DNN per dataset 1.000.000 jets (```MC23G_dataset```) ridotto a 100 feature

### Risultati

**Obiettivo 2.2:**

- **PRELIMINARE - Procedura corretta di standardizzazione delle feature tra trainset, validation set e testset**

    La procedura utilizzata in fase di preprocessing per la standardizzazione dei dataset:

    ```python
    mean = np.mean(X_train, axis=0)
    std  = np.std(X_train, axis=0)

    X_train = (X_train - mean) / std
    X_val   = (X_val - mean) / std
    X_test  = (X_test - mean) / std
    ```

    dove i pesi per la normalizzazione sono calcolati **unicamente sul trainset**, e poi riutilizzati per standardizzare anche validation e testset, sembra essere lo standard considerato corretto in Machine Learning.

    Di seguito alcune fonti:

    - In [scikit-learn User Guide: Common Pitfalls and Recommended Practices](https://scikit-learn.org/stable/common_pitfalls.html?highlight=linearregression&utm_source=chatgpt.com) si raccomanda di effettuare la standardizzazione sia su trainset che su testset, e nell'esempio riportato i pesi utilizzati per lo scaling del testset sono quelli ottenuti precedentemente sul trainset:

      ```python
      from sklearn.metrics import mean_squared_error
      from sklearn.linear_model import LinearRegression
      from sklearn.preprocessing import StandardScaler

      scaler = StandardScaler()
      X_train_transformed = scaler.fit_transform(X_train)
      model = LinearRegression().fit(X_train_transformed, y_train)
      mean_squared_error(y_test, model.predict(X_test))
      ```

      **Right:** Instead of passing the non-transformed `X_test` to `predict`, we should transform the test data, the same way we transformed the training data:

      ```python
      X_test_transformed = scaler.transform(X_test)
      mean_squared_error(y_test, model.predict(X_test_transformed))
      0.90...
      ```

      dove (si veda [scikit-learn Documentation - StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html?utm_source=chatgpt.com)) il metodo `fit_transform(X_train)` calcola i pesi per la normalizzazione sul trainset (fit) e li applica allo stesso dataset (transform), mentre `transform(X_test)` usa **gli stessi pesi** ottenuti sul trainset per riscalare anche il testset.

    - Una ulteriore discussione sul tema si trova anche su [questo forum](https://stats.stackexchange.com/questions/522592/data-leakage-concerns):

      > Normalisation/standardisation is also a part of the modelling process, and it should be completed before testing. Imagine your model is deployed and making predictions online (e.g. on user devices), you won't have access to aggregate test statistics because the test set is distributed across many devices online, so you use training statistics. The purpose of the test set is to emulate this behaviour so that you can have an idea of how your model will perform when it's deployed. Using test set statistics in your evaluation wouldn't be fair since you won't be able to do that in runtime.


- **OSSERVAZIONE - Distribuzione delle feature nel dataset ```MC23G``` (pt.2)**:
Analizzando le proprietà statistiche associate alle singole feature precedentemente selezionate:
    ```python
    [
        "d0_trk_i",
        "z0SinTheta_trk_i",
        "dphi_trk_i",
        "deta",
        "pt",
        "lifetimeSignedD0Significance",
        "lifetimeSignedZ0SinThetaSignificance",
        "qOverP",
        for i in range(len(num_tracks_per_jet))
    ]
    ```

    con ```(num_tracks_per_jet)=20```, si sono individuate le feature che presentano maggiore deviazioni dai valori mean=0, std=1 su validation set e testset dopo aver effettuato la standardizzazione con i pesi associati al trainset; rimuovendo dai dataset le seguenti feature individuate:
    ```python
    [
        "pt",
        "lifetimeSignedD0Significance",
        "lifetimeSignedZ0SinThetaSignificance"
    ]
    ```
    si ottengono statistiche su validatione e test set che si discostano significativamente meno dalla distribuzione normalizzata attesa dopo il procedimento di standardizzazione rispetto ai risultati anomali precedentemente ottenuti:
    ```text
    TRAIN -> mean range [-0.000000, 0.000000] | std range [1.000000, 1.000000]
    VAL   -> mean range [-0.011486, 0.000656] | std range [0.000059, 1.431836]
    TEST  -> mean range [-0.008927, 0.000873] | std range [0.000059, 1.431846]
    ```
    Si procede con l'addestramento di una rete utilizzando questo dataset con un numero di feature ridotte rispetto alla versione precedente, in particolare la DNN avrà in input **((8-3) * 20) = 100** feature.
    
    
**OUTPUT - Addestramento della DNN per trainset di 1.000.000 jets (dataset `MC23G`)**:
    Testata performance del training per:
        - **Hidden layers**: `(64, 32, 16)`
        - **Dimensiona batch**: `16384`
        - **Dropout**: `0.3`
    al variare del tipo di normalizzazione sui layer:
        - [run con ```BatchNorm1D```](https://github.com/giumont/flavour_tagging/tree/main/outputs/MC23G_dataset/DNN/1e6_reduced/3layers_64_32_16_dropout0.3_BatchNorm1d_batch16384)
        - [run con ```LayerNorm```](https://github.com/giumont/flavour_tagging/tree/main/outputs/MC23G_dataset/DNN/1e6_reduced/3layers_64_32_16_dropout0.3_LayerNorm_batch16384)
    
   
 Per queste scelte di parametri, la normalizzazione su batch (`BatchNorm1d`) risulta più efficace rispetto a `LayerNorm`:

    | Metrica (test set)                      | BatchNorm1d  |  LayerNorm   |
    | ----------------------------------------|:------------:|:------------:|
    | best_val_loss                           |    0.4852    |    0.5380    |
    | epoche effettive (early stop)           | 50 (best@48) | 35 (best@25) |
    | AUC su testset                          |    0.8499    |    0.8195    |
    | signal eff. @ soglia score 0.5          |    73.20%    |    76.99%    |
    | background rejection @ soglia score 0.5 |     5.42     |     3.75     |

**Working points a parità di signal efficiency (test set)** (-> a parità di soglia per lo score):

    | target eff. | bkg rejection BatchNorm1d | bkg rejection LayerNorm |
    |:-----------:|:-------------------------:|:------------------------:|
    | 60%         | 9.08                       | 6.40                     |
    | 70%         | 6.16                       | 4.73                     |
    | 77%         | 4.62                       | 3.75                     |
    | 85%         | 3.12                       | 2.76                     |
    | 90%         | 2.37                       | 2.20                     |

**Commenti:**
    - `BatchNorm1d` raggiunge una `best_val_loss` significativamente più bassa (0.485 vs 0.538); il run con `LayerNorm` va in early stopping molto prima (epoca 35, best model @25) rispetto a `BatchNorm1d` (epoca 50, best model @48): probabilmente `batch_size` grande -> statistiche di normalizzazione stabilil.
    - A parità di signal efficiency, `BatchNorm1d` fornisce una background rejection sistematicamente superiore su tutto il range testato (60–90%), suggerendo una separazione segnale/fondo più netta a monte della scelta di soglia.
    - Per questo dataset, a differenza di ```salt-tutorial```, dopo la riduzione a 100 feature **non si osserva più la loss crescente sul validation set** quanto si utilizza ```BatchNorm1D```, ottenendo i comportamenti standard attesi.


---
## Ven 2026-09-11

### Attività

**Obiettivo 2.2:**
- Riorganizzazione del codice del progetto:
    - Funzioni riscritte per dipendere esclusivamente dai parametri globali specificati nel notebook di training
    - Salvataggio delle run di training e evaluation: funzioni per creazione automatica del file su repo di github:
        - ```config.yaml``` (parametri di configurazione della run per riproducibilità)
        - grafici
        - file `.txt` con parametri di valutazione del modello 

 
### Risultati

**OUTPUT - Addestramento della DNN per trainset di ~9.000.000 jets (dataset `MC23G`) - pt.1**:

Testata performance del training per:

- Reti con maggiore espressività rispetto al best model con trainset 1e6:
    - **Hidden layers**: **`(128, 64, 32), (256, 128, 64)`**
    - **Dimensione batch**: **`65536`**
    - **Dropout**: `0.2` o `0.3` 
    - **Normalizzazione per layer**: ```BatchNorm1D``` 
    - **LR**: **`4e-3`**

    ![](https://codimd.web.cern.ch/uploads/upload_8a48b94520ef119d06a81d51d098b832.png)
    ![](https://codimd.web.cern.ch/uploads/upload_f567d974084eb06eab4f0e005c3ee9c8.png)
    ![](https://codimd.web.cern.ch/uploads/upload_3103a78e14f514c68c6ecac1c5bb7570.png)


    **NOTA**: si noti l'aumento della dimensione del batch rispetto alle run precedenti; visto l'utilizzo di ```BatchNorm1D```, si presumeva che questo potesse **stabilizzare le statistiche su cui è calcolata la normalizzazione** e **velocizzare la convergenza nel training**; **rimane però un punto da approfondire e non si esclude possa aver influito (negativamente) sui risultati ottenuti**

    **NOTA**: si noti l'aumento del learning rate (LR) rispetto alle run precedenti (best model: 3e-3); **questo cambiamento non è sufficientemente giustificato e potrebbe aver influito (negativamente) sui risultati ottenuti**

 
- [Run](https://github.com/giumont/flavour_tagging/tree/main/outputs/MC23G_dataset/DNN/full_reduced/3layers_64_32_16_dropout0.3_BatchNorm1d_batch16384) con rete con stessa architettura del best model per il [trainset da 1e6 jets](https://github.com/giumont/flavour_tagging/tree/main/outputs/MC23G_dataset/)
         - **Hidden layers**: `(64, 32, 16)`
        - **Dimensiona batch**: `16384`
        - **Dropout**: `0.3`

     Per queste scelte di parametri, le performance del modello **non sono significativamente migliori rispetto al precedente best case su trainset di 1e6 jet**. In particolare:

        | Metrica (test set)                      | Trainset 1e6 jet (best model)  |  Trainset ~9e6 jet   |
        | ----------------------------------------|:------------:|:------------:|
        | best_val_loss                           |    0.4852    |    0.4893    |
        | epoche effettive (early stop)           | 50 (best@48) | 25 (best@15) |
        | AUC su test set                          |    0.8499    |    0.8469    |
        | signal eff. @ soglia score 0.5          |    73.20%    |    76.15%    |
        | background rejection @ soglia score 0.5 |     5.42     |     4.62     |

    **Working points a parità di signal efficiency (test set)** (-> a parità di soglia per lo score):

        | target eff. | bkg rejection  1e6 jet (best)    | bkg rejection ~9e6 jet |
        |:-----------:|:-------------------------:|:------------------------:|
        | 60%         | 9.08                       | 8.27                     |
        | 70%         | 6.16                       | 5.82                     |
        | 77%         | 4.62                       | 4.47                     |
        | 85%         | 3.12                       | 3.14                     |
        | 90%         | 2.37                       | 2.40                     |
    I valori di best validation loss sono molto simili per le due run, e in entrambi casi la loss sembra assestarsi attorno ad essi (plateau)
                - Trainset 1e6 jet (best model):
                ![](https://codimd.web.cern.ch/uploads/upload_f919ab02b189bb6a7bf49e4ca51a2aa8.png)
                - Trainset ~9e6 jet:
                ![](https://codimd.web.cern.ch/uploads/upload_b0c6297d13b1d23f8cc9ff9568aa3933.png)


    **NOTA**: [La run che è stata effettuata anche con ```LayerNorm``` al posto di ```BatchNorm```](https://github.com/giumont/flavour_tagging/tree/main/outputs/MC23G_dataset/DNN/full_reduced/3layers_64_32_16_dropout0.3_LayerNorm_batch16384) ha prodotto performance peggiori rispetto a quella sopra. 


    **Commenti**
    - Nelle diverse run, la loss sul validation set sembra assestarsi sempre attorno a `~0.5`: modificare l'espressività dell'architettura (mantenendo i tre layer) sembra non influenzare particolarmente questo risultato, determinando al massimo la velocità di convergenza o le oscillazioni attorno a questa soglia (*il modello non sta davvero imparando la fisica, trasferibile a dataset mai visti?*)
    - La loss sul train set scende di più per architetture con più neuroni negli hidden layers (`~0.35` VS `~0.40`), ma poichè questo risultato non è accompagnato da una discesa della loss sul validation set, è un segnale che aumentare il numero di parametri addestrabili è già fonte di ulteriori overfitting, nelle configurazioni provate
-> **bisognerebbe provare ad abbinare forme di regolarizzazione / dropout più forti / modificare LR per indagare se questo migliora il tradeoff tra le due loss**


---
## Sab 2026-09-12

### Attività

**Obiettivo 2.2**
    - Run dataset `MC23G`
    - Studio del regime energetico associato al dataset `MC23G`
    - Studio dell'ordinamento delle tracce nel dataset `MC23G`
    - Plotting delle distribuzioni di alcuni parametri cinematici per le tracce nel dataset `MC23G` e confronto con `salt-tutorial`

### Risultati

- **OSSERVAZIONE**
- **OSSERVAZIONE - Il dataset `MC23G` è per alti valori di $p_T$:**
  Per determinare il corretto benchmark di confronto (regime $t\bar{t}$ (*non-boosted*) oppure al regime $Z'$ (*boosted*) del paper [Transforming jet flavour tagging at ATLAS](https://arxiv.org/abs/2505.19689)) per i modelli allenati sul dataset `MC23G`, è stata condotta un'indagine cinematica sulla
<!--  
 
    #### Definizione Fisica dei Regimi Cinematici
    La distinzione tra regime *boosted* e *non-boosted* è definita dall'impulso trasverso ($p_T$) dei jet e dall'effetto del boost relativistico sui loro costituenti:
    - **Regime Non-Boosted ($t\bar{t}$):** Caratterizzato da jet con $p_T < 250\text{ GeV}$[cite: 10]. A queste energie, i prodotti del decadimento adronico sono spazialmente ben separati all'interno dell'intero cono del jet ($R = 0.4$)[cite: 11].
    - **Regime Boosted ($Z'$):** Caratterizzato da jet ad altissima energia con $p_T \gg 250\text{ GeV}$ (fino alla scala dei $\text{TeV}$)[cite: 8, 13]. Il forte boost di Lorentz comprime angolarmente i prodotti di decadimento attorno all'asse del jet secondo la relazione cinematica $\Delta R \approx \frac{2 m}{p_T}$.

    #### Osservabili e Indagini Condotte
    Per verificare la natura del dataset, sono state analizzate due osservabili fondamentali: lo **spettro di $p_T$** e la **distanza angolare relativa ($\Delta R$)**:
    1. **Spettro di $p_T$ delle tracce:**
       - Il dataset di controllo `salt-tutorial` presenta un cutoff rigido intorno a $p_T \approx 250\text{ GeV}$[cite: 10], confermando la sua natura di campione puramente $t\bar{t}$.
       - Il dataset `MC23G` presenta invece una coda ad altissima energia che si estende fino a $p_T \approx 4.6 \cdot 10^6\text{ MeV}$ ($4.6\text{ TeV}$)[cite: 8, 13], evidenziando una popolazione dominata da eventi ad altissimo impulso trasverso.
       [IMMAGINI] -->
 
 **Distanza angolare $\Delta R = \sqrt{(\Delta\eta)^2 + (\Delta\phi)^2}$:**
La variabile $\Delta R$ misura la distanza geometrica delle tracce rispetto all'asse principale del jet; vale la relazione cinematica:  $\Delta R \approx \frac{2 m}{p_T}$.
       - Nel dataset `salt-tutorial`, la distribuzione di $\Delta R$ è diffusa su tutto il cono del jet fino a $\Delta R = 0.4$, con una densità ridotta a brevi distanze.
       ![](https://codimd.web.cern.ch/uploads/upload_76b6a1979b48acbb64ec1562dbedfb9b.png)
     - Nel dataset `MC23G`, la distribuzione mostra un picco a $\Delta R \approx 0.04$, al di sotto della soglia convenzionale di collimazione ($\Delta R = 0.05$). Tale concentrazione spaziale nel *core* del jet costituisce la prova geometrica della presenza di jet fortemente *boosted*.
       ![](https://codimd.web.cern.ch/uploads/upload_bd8c58a01932bcddca7f6889a9107c9b.png)

Quanto osservato è coerente con la documentazione [ATLAS Flavour Tagging Documentation](https://ftag.docs.cern.ch/samples/h5_sample_list/#centrally-preprocessed-h5-samples-for-general-trainings) (il dataset `MC23G` è etichettato come "Extended $Z'$ with Gluons & Hadronic Taus")



**OUTPUT - Addestramento della DNN per trainset di ~9.000.000 jets (dataset `MC23G`) - pt.2**:
Testata performance del training per tre modifiche agli iperparametri rispetto al [best case attuale per questa dimensione del dataset](https://github.com/giumont/flavour_tagging/tree/main/outputs/MC23G_dataset/DNN/full_reduced/3layers_64_32_16_dropout0.3_BatchNorm1d_batch16384) (si veda pt.1)

- [Stessa architettura, con ```LayerNorm``` al posto di ```BatchNorm1D```](https://github.com/giumont/flavour_tagging/tree/main/outputs/MC23G_dataset/DNN/full_reduced/3layers_64_32_16_dropout0.3_LayerNorm_batch16384):
        - **Hidden layers**: `(64, 32, 16)`
        - **Dimensione batch**:`16384` 
        - **Dropout**: `0.3` 
        - **Normalizzazione per layer**: **```LayerNorm```** 
        
    ![](https://codimd.web.cern.ch/uploads/upload_5ef8026551f9ee43ced8f73f3e662b44.png)
    
    NOTA: in questo caso la dimensione del batch potrebbe aver influito sulla velocità del training, ma non entra nel processo di normalizzazione

- [Architettura con più hidden layers, e mantenendo ```BatchNorm1D```](https://github.com/giumont/flavour_tagging/tree/main/outputs/MC23G_dataset/DNN/full_reduced/4layers_256_256_128_64_dropout0.2_BatchNorm1d_batch65536):
        - **Hidden layers**: **`(256, 256, 128, 64)`**
        - **Dimensione batch**: `65536`
        - **Dropout**: `0.2`
        - **Normalizzazione per layer**: ```BatchNorm1D``` 

    Tentativo fallimentare: discesa della loss sul trainset effettivamente velocizzata, ma peggiora overtraining e generalizzazione
        ![](https://codimd.web.cern.ch/uploads/upload_73ac8e2ee5cea90e819284cf8104cab3.png)

    **NOTA**: Abbassare il dropout rispetto ai casi di confronto (`0.3`) potrebbe essere stato un fattore confondente in questa run e aver estremizzato l'overfitting
    
**Commenti:**
    - Ancora una volta, i valori di validation loss non scendono sotto `~0.5`
    - Aggiungere piu layer sembra peggiorare il problema dell'overfitting, ma **rimane da testare se la situazione cambia aggiungendo maggiore regolarizzazione / aumentando il dropout**

---
## Dom 2026/09/13

### Attività

- [DA COMPLETARE] Preparazione dataset `MC23G` per training:
    - Retrieve di pT, S(d0) e S(z0*sintheta) dai file .h5 per stessi jets (indici salvati in precedente preprocessing) selezionati per trainset, validation set e testset per le altre features considerate
    - Preprocessing: implementata bozza di cut sugli outlier per queste tre features, basato sulle statistiche del trainset
    - Implementato codice per appendere ai dataset .npy e .npz precedentemente creati per le vecchie features i dati sulle nuove features aggiunte

- Nuove run (senza features aggiuntive) per `MC23G`

### Risultati

** PRELIMINARE - Aggiunta features in input per il training su `MC23G`**
[AGGIUNGI]
 
**OUTPUT - Addestramento della DNN per trainset di ~9.000.000 jets (dataset `MC23G`) - pt.3**:
Provato training per:
- Architetture con 4 hidden layers [come quella già provata](https://github.com/giumont/flavour_tagging/tree/main/outputs/MC23G_dataset/DNN/full_reduced/4layers_256_256_128_64_dropout0.2_BatchNorm1d_batch65536), ma [per batch più piccoli](https://github.com/giumont/flavour_tagging/tree/main/outputs/MC23G_dataset/DNN/full_reduced/4layers_256_256_128_64_dropout0.2_BatchNorm1d_batch4096) / [con `LayerNorm` al posto di `BatchNorm1D`](https://github.com/giumont/flavour_tagging/tree/main/outputs/MC23G_dataset/DNN/full_reduced/4layers_256_256_128_64_dropout0.2_LayerNorm_batch4096):
        - **Hidden layers**: `(256, 256, 128, 64)`
        - **Dimensione batch**: **`4096`** 
        - **Dropout**: `0.2` 
        - **Normalizzazione per layer**: `BatchNorm1D` o **```LayerNorm```** 
*Motivazione*: i risultati patologici sulle validation loss "piatte" possono essere dovute ad un comportamento di `BatchNorm1D` + scelta infelice della dimensione del batch, o a `BatchNorm1D` stesso?
    - Con ```BatchNorm1D``` e batch ridotto:
    ![](https://codimd.web.cern.ch/uploads/upload_cc942650848aa577b40492efebaf778a.png)
    - Con ```LayerNorm``` e batch ridotto:
    ![](https://codimd.web.cern.ch/uploads/upload_35dc4f1b1a17e0b4914ceb7f04204233.png)



- [BOZZA][Architettura con (ancora più) hidden layers, ma con ``LayerNorm``](https://github.com/giumont/flavour_tagging/tree/main/outputs/MC23G_dataset/DNN/full_reduced/6layers_512_512_256_256_128_64_dropout0.2_LayerNorm_batch65536):
        - **Hidden layers**: **`(512, 512, 256, 128, 64)`**
        - **Dimensione batch**: **`65536`** 
        - **Dropout**: **`0.2`** 
        - **Normalizzazione per layer**: **```LayerNorm```** 
*Motivazione*: La loss sul trainset aumentando l'espressività del modello scende oltre le soglie osservate per le run precedenti a parità di epoche? Usare `LayerNorm` mitiga l'overtraining che ci aspetteremmo sul validation set?

    ![](https://codimd.web.cern.ch/uploads/upload_8998c0b1e1d2d8c13159572dd5e11fbc.png)

        | Metrica (test set)                      | Trainset 1e6 jet (best model) | Trainset ~9e6 jet (best model @1e6 jet) | 6layers LayerNorm (~9e6 jet) |
        | ----------------------------------------|:----------------------------:|:-----------------:|:---------------------------:|
        | best_val_loss                            |            0.4852            |      0.4893       |           0.5295            |
        | epoche effettive (early stop)           |         50 (best@48)         |   25 (best@15)    |        39 (best@28)         |
        | AUC su test set                         |            0.8499            |      0.8469       |           0.8264            |
        | signal eff. @ soglia score 0.5          |            73.20%            |      76.15%       |           76.96%            |
        | background rejection @ soglia score 0.5 |             5.42             |       4.62        |            3.75             |



 **Commenti**

 - Chiarificato che il problema (stavolta, a differenza di `salt-tutorial`) sembra indipendente dalla scelta di `BatchNorm1D` / `LayerNorm`
 - Si rafforza l'ipotesi che **il problema sia nella scelta delle feature** (non abbastanza discriminanti?) e/o **in una divergenza tra il train set e il validation set** -> "il modello non impara la fisica, per questo non generalizza"


---
## Lun 2026/09/14

### Attività
- Preparazione slides incontro online (riepilogo quanto fatto finora) martedi 15/09

### Risultati
--

---
## Mar 2026/09/15

### Attività

**Obiettivo 2.2**
- Approfondimento su regime $Z'$ e dipendenza dal boost del parametro di impatto $d_0$

**Obiettivo 3.1**
- Studio preliminare della struttura dei file ROOT eventi $HH\to bb\tau\tau$

**Altro**
- Incontro online: riepilogo quanto fatto finora + discussione prossimi obiettivi (punto 3.)
- Update e ordine del logbook su codimd



### Risultati

**APPROFONDIMENTO - Comparabilità del parametro di impatto $d_0$ tra diversi regimi cinematici ($t\bar  t$ e $Z'$)**

Il parametro di impatto trasverso $d_0$ si dimostra essere, in regime ultrarelativistico ($\gamma \gg1, \beta \to 1$), **asintoticamente indipendente dal boost di Lorentz della particella madre** $\gamma$.

Infatti, per definizione $d_0$ è la minima distanza della traccia dal punto di interazione primario (IP) nel piano trasverso:
$$
d_0 = L_{xy} \cdot \sin(\Delta \phi)
$$
dove:
- $L_{xy}$ è la lunghezza di volo nel piano trasverso della particella madre (adrone B nel caso di b-jet) (in figura: $\lambda$).
In termini di lunghezza di volo della particella madre:
$$
L_{xy} = L\sin\theta = \beta \gamma c \tau_0 \sin\theta
$$
dove $\theta$ è l'angolo polare della particella madre rispetto all'asse $z$ (asse del campo magnetico in ATLAS) e $\tau_0$ è il suo tempo proprio di decadimento
- $\Delta\phi = \phi_{madre} - \phi_{traccia}$ è l'angolo di apertura azimutale (nel piano trasverso) tra la particella madre e la traccia (in figura: $\Psi$)
![](https://codimd.web.cern.ch/uploads/upload_0e901c5a638d955ab70064b9bffc585b.png)

    In regime ultrarelativistico:
    $$
    \sin\Delta\phi = \frac {p_T}E = \frac{p^*_T}{\gamma (E^*+\beta p^*_\parallel)} \stackrel{\beta\to 1} {\to}\frac 1\gamma \frac{\sin \Delta\phi^*}{1+\cos\Delta\phi^*} = \frac 1\gamma f(\Delta\phi^*)
    $$
    da cui per $\beta\to 1$:
    $$
    d_0 \approx \gamma c\tau_0 \cos\theta \frac 1\gamma f(\Delta\phi^*) = c\tau_0f(\Delta\phi^*)\cos\theta
    $$
    indipendente da $\gamma$.

    Quindi nel regime boosted associato a eventi $Z'$ ($p_T \gg m$) si può fare questa approssimazione, e questo rende la distribuzione di $d_0$ dipendente solo dall'invariante relativistico $c\tau_0$, dunque comparabile con quella per eventi $t\bar t$ a energie più basse. 
    


**PRELIMINARE - Struttura dei file `.root` associati agli eventi $HH\to bb\tau\tau$**

I file forniti si presentano come un unico `TTree` denominato `AnalysisMiniTree` con i seguenti branches:

* `[event_misc]`: Metadati evento, pile-up, pesi, MET, SF globali 
    * **Livello**: Event
    * **Stato**: Certo [Classificato da AI, in sospeso]

* `[trigPassed_]`: Decisioni booleane di HLT chain specifiche (jet multipli, tau)
    * **Livello**: Trigger Level
    * **Stato**: Certo [Classificato da AI, in sospeso]

* `[el_]`: Elettroni ricostruiti

  | Nome variabile | Formato | Livello | Descrizione | Stato |
  |---|---|---|---|---|
  | `el_pt___NOSYS`, `el_eta`, `el_phi` | `std::vector<float>` (Jagged) | Reco Level | Cinematica dell'elettrone ricostruito | Certo |
  | `el_charge` | `std::vector<float>` (Jagged) | Reco Level | Carica elettrica dell'elettrone | Certo |
  | `el_d0`, `el_d0sig`, `el_z0sintheta`, `el_z0sinthetasig` | `std::vector<float>` (Jagged) | Reco Level | Variabili di parametro d'impatto trasverso e longitudinale con relative significatività | Certo |
  | `el_isAnalysisElectron___NOSYS` | `std::vector<bool>` (Jagged) | Analysis Level | Flag di selezione dell'elettrone per l'analisi | Certo |
  | `el_isLepton1___NOSYS`, `el_isLepton2___NOSYS` | `std::vector<bool>` (Jagged) | Analysis Level | Flag di assegnazione all'oggetto `bbtt_Lepton1`/`Lepton2` a livello analisi | In corso di verifica |
  | `el_parentHiggsParentsMask`, `el_parentTopParentsMask`, `el_parentZParentsMask` | `std::vector<unsigned int>` (Jagged) | Truth Level (Matching) | Maschere a bit di provenienza truth (identificazione della particella madre Higgs, Top, Z) | In corso di verifica |

* `[mu_]`: Muoni ricostruiti (struttura analoga a `el_`)

  | Nome variabile | Formato | Livello | Descrizione | Stato |
  |---|---|---|---|---|
  | `mu_pt___NOSYS`, `mu_eta`, `mu_phi` | `std::vector<float>` (Jagged) | Reco Level | Cinematica del muone ricostruito | Certo |
  | `mu_charge` | `std::vector<float>` (Jagged) | Reco Level | Carica elettrica del muone | Certo |
  | `mu_d0`, `mu_d0sig`, `mu_z0sintheta`, `mu_z0sinthetasig` | `std::vector<float>` (Jagged) | Reco Level | Variabili di parametro d'impatto trasverso e longitudinale con relative significatività | Certo |
  | `mu_isAnalysisMuon___NOSYS` | `std::vector<bool>` (Jagged) | Analysis Level | Flag di selezione del muone per l'analisi | Certo |
  | `mu_isLepton1___NOSYS`, `mu_isLepton2___NOSYS` | `std::vector<bool>` (Jagged) | Analysis Level | Flag di assegnazione all'oggetto `bbtt_Lepton1`/`Lepton2` a livello analisi | In corso di verifica |
  | `mu_parentHiggsParentsMask`, `mu_parentTopParentsMask`, `mu_parentZParentsMask` | `std::vector<unsigned int>` (Jagged) | Truth Level (Matching) | Maschere a bit di provenienza truth (identificazione della particella madre Higgs, Top, Z) | In corso di verifica |

* `[tau_]`: Tau adronici ricostruiti

  | Nome variabile | Formato | Livello | Descrizione | Stato |
  |---|---|---|---|---|
  | *`tau_decayMode`, `tau_nProng`* | `std::vector<int>` (Jagged) | Reco Level | Modo di decadimento del tau e numero di tracce adroniche (prong) | Da verificare |
  | ***`tau_GNTauScoreSigTrans_v0prune`***| `std::vector<float>` (Jagged) | Reco / Analysis Level | Score del tagger per identificazione del tau basato su rete neurale GNN | Da verificare |
  | **`tau_RNNEleScoreSigTrans_v1`** |  `std::vector<float>` | |
  | **`tau_RNNJetScoreSigTrans`**| ` std::vector<float>` | |
  | `tau_pt___NOSYS`, `tau_eta`, `tau_phi`, `tau_charge` | `std::vector<float>` (Jagged) | Reco Level | Cinematica e carica del tau adronico ricostruito | Certo |
  |**`tau_isAnalysisTau___NOSYS`** | `std::vector<bool>` (Jagged) | Analysis Level | Flag di selezione del tau per l'analisi | Certo |
  | **`tau_isTau1___NOSYS`, `tau_isTau2___NOSYS`** | `std::vector<bool>` (Jagged) | Analysis Level | Flag di assegnazione all'oggetto `bbtt_Tau1`/`Tau2` a livello analisi | Certo |
  | ***`tau_truthType`, `tau_truthOrigin`, `tau_tauTruthJetLabel`*** | `std::vector<int>` (Jagged) | Truth Level (Matching) | Informazioni di matching e classificazione a livello truth per il tau | Da verificare |
  | **`tau_truth_IsHadronicTau`** | `std::vector<bool>` (Jagged) | Truth Level | Flag booleano che indica se l'oggetto è un vero tau adronico a livello truth | Certo |
  | ***`tau_parentHiggsParentsMask`, `tau_parentTopParentsMask`, `tau_parentZParentsMask`*** | `std::vector<unsigned int>` (Jagged) | Truth Level (Matching) | Maschera a bit di provenienza truth (particella madre Higgs, Top, Z) | In corso di verifica |

* `[recojet_antikt4PFlow_]`: Jet ricostruiti anti-kt R=0.4 PFlow

  | Nome variabile | Formato | Livello | Descrizione | Stato |
  |---|---|---|---|---|
  | `recojet_antikt4PFlow_pt___NOSYS`, `recojet_antikt4PFlow_eta`, `recojet_antikt4PFlow_phi`, `recojet_antikt4PFlow_m___NOSYS` | `std::vector<float>` (Jagged) | Reco Level | Cinematica e massa del jet ricostruito | Certo |
  | ***`recojet_antikt4PFlow_ftag_quantile_GN2v01_Continuous`*** | `std::vector<float>` (Jagged) | Analysis Level | Score continuo del tagger `GN2v01` espresso come quantile | Da verificare|
  | **`recojet_antikt4PFlow_ftag_select_GN2v01_FixedCutBEff_85`** | `std::vector<bool>` (Jagged) | Analysis Level | Decisione di b-tagging pass/fail al working point a efficienza fissa dell'85% | Certo |
  | *`recojet_antikt4PFlow_ftag_effSF_GN2v01_Continuous___NOSYS`* | `std::vector<float>` (Jagged) | Analysis Level | Scale Factor di efficienza per b-tagging continuo | In corso di verifica |
  | **`recojet_antikt4PFlow_isbjet1___NOSYS`**, **`recojet_antikt4PFlow_isbjet2___NOSYS`** | `std::vector<bool>` (Jagged) | Analysis Level | Flag di assegnazione ai jet `bbtt_Jet_b1`/`b2` selezionati dall'algoritmo | Certo |
  | **`recojet_antikt4PFlow_isAnalysisJet___NOSYS`** | `std::vector<bool>` (Jagged) | Analysis Level | Flag di selezione del jet per l'analisi | Certo |
  | ***`recojet_antikt4PFlow_HadronConeExclTruthLabelID`*** | `std::vector<int>` (Jagged) | Truth Level | Flavour truth del jet basato sugli adroni nel cono (b/c/light/tau) | In corso di verifica |
  | ***`recojet_antikt4PFlow_bJetTruthDR`, `recojet_antikt4PFlow_bJetTruthPt`*** | `std::vector<float>` (Jagged) | Truth Level | Distanza angolare Delta R e pt del b-quark truth associato al jet | Da verificare |
  | *`recojet_antikt4PFlow_nTopToBChildren`, `recojet_antikt4PFlow_nTopToWChildren`* | `std::vector<int>` (Jagged) | Truth Level | Numero di figli truth provenienti da decadimenti di Top in b o W | Da verificare |
  | ***`recojet_antikt4PFlow_parentHiggsParentsMask`, `recojet_antikt4PFlow_parentScalarParentsMask`, `recojet_antikt4PFlow_parentTopParentsMask`, `recojet_antikt4PFlow_parentZParentsMask`*** | `std::vector<unsigned int>` (Jagged) | Truth Level (Matching) | Maschere a bit di provenienza truth per il matching jet-particella madre | In corso di verifica |

* `[bbtt]`: Livello analisi: oggetti/coppie già selezionate dall'algoritmo di ricostruzione $HH\to bb\tau\tau$

  | Nome variabile | Formato | Livello | Descrizione | Stato |
  |---|---|---|---|---|
  | **`bbtt_Jet_b1_*`**, **`bbtt_Jet_b2_*`** | `float`, `int` (Scalari per evento / Oggetti) | Analysis Level | Variabili dei due b-jet selezionati (`pt`, `eta`, `phi`, `E`, `uncorrPt`, `muonCorrPt`, `nmuons`) | Certo | 
  | **`bbtt_Jet_b1_truthLabel`**, **`bbtt_Jet_b2_truthLabel`** | | |
  | **`bbtt_Jet_pcbt_GN2v01`**, **`bbtt_Jet_b2_pcbt_GN2v01`** | | |
  | **`bbtt_Tau1_*`**, **`bbtt_Tau2_*`** | `float`, `int`, `bool` (Scalari per evento / Oggetti) | Analysis Level | Variabili dei due tau selezionati (`pt`, `eta`, `phi`, `E`, `RNN`, `decayMode`, `nProng`, `charge`, `isTauID`, `isAntiTau`, `EleRNN_WP`, `truthType`, `tauTruthJetLabel`) | Certo |
  | **`bbtt_Tau1_truthType`**, **`bbtt_Tau2_truthType`** | | |
  | **`bbtt_Tau1_tauTruthJetLabel`**, **`bbtt_Jet_tauTruthJetLabel`** | | |
  | `bbtt_Lepton1_*`, `bbtt_Lepton2_*` | `float`, `int`, `bool` (Scalari per evento / Oggetti)   | Analysis Level | Variabili dei leptoni selezionati nei canali LepHad/DiLep (`pt`, `eta`, `phi`, `E`, `charge`, `pdgid`, `isIso`, `effSF`) | Certo |
  | `bbtt_H_bb_pt`, `bbtt_H_bb_eta`, `bbtt_H_bb_phi`, `bbtt_H_bb_m` | `float` (Scalari per evento) | Analysis Level | Quadrimpulso e massa del sistema Higgs decay in b-bar ricostruito dai due b-jet | Certo |
  | *`bbtt_H_vis_tautau_pt`, `bbtt_H_vis_tautau_eta`, `bbtt_H_vis_tautau_phi`, `bbtt_H_vis_tautau_m`* | `float` (Scalari per evento) | Analysis Level | Quadrimpulso e massa del sistema Higgs decay in tau-tau visibile (senza neutrini/MMC) | In corso di verifica |
  | `bbtt_HH_pt`, `bbtt_HH_eta`, `bbtt_HH_phi`, `bbtt_HH_m` | `float` (Scalari per evento) | Analysis Level | Quadrimpulso e massa del sistema HH completo ricostruito (con MMC) | Certo |
  | *`bbtt_HH_vis_pt`, `bbtt_HH_vis_eta`, `bbtt_HH_vis_phi`, `bbtt_HH_vis_m`* | `float` (Scalari per evento) | Analysis Level | Quadrimpulso e massa visibile del sistema HH | Da verificare |
  | **`bbtt_HH_delta_phi`, `bbtt_HH_vis_delta_phi`** | `float` (Scalari per evento) | Analysis Level | Separazione angolare Delta-phi tra il sistema b-bar e il sistema tau-tau (totale e visibile) | Certo |
  | *`bbtt_mmc_m`, `bbtt_mmc_pt`, `bbtt_mmc_eta`, `bbtt_mmc_phi`* | `float` (Scalari per evento) | Analysis Level | Massa invariante e cinematica del sistema tau-tau ricostruita con Missing Mass Calculator | In corso di verifica |
  | *`bbtt_mmc_status`* | `int` (Scalare per evento) | Analysis Level | Stato di convergenza dell'algoritmo MMC (0/1/2 = fallito/riuscito/degenerazione) | In corso di verifica |
  | **`bbtt_pass_presel___NOSYS`** | `int8_t` (Scalare per evento) | Analysis Level | Flag di superamento della preselezione di base | Certo |
  | *`bbtt_pass_SR_1B___NOSYS`*, *`bbtt_pass_SR_2B___NOSYS`* | `bool` (Scalari per evento) | Analysis Level | Flag di selezione per la Signal Region con 1 o 2 b-jet taggati | Da verificare |
  | ***`bbtt_pass_HadHad___NOSYS`, `bbtt_pass_HadHad_1B___NOSYS`, `bbtt_pass_HadHad_2B___NOSYS`***| `bool` (Scalari per evento) | Analysis Level | Flag di selezione per il canale completamente adronico (tau_had tau_had) | Da verificare |
<!--
  | `bbtt_mmc_nu1_*`, `bbtt_mmc_nu2_*` | `float` (Scalari per evento) | Analysis Level | Stima dei quadrimpulsi dei due neutrini prodotti dall'algoritmo MMC | In corso di verifica |
  | `bbtt_eventTriggerSF___NOSYS` | `float` (Scalare per evento) | Analysis Level | Scale Factor di trigger complessivo a livello di evento | Certo |
  | `bbtt_pass_ZCR___NOSYS`, `bbtt_pass_TopEMuCR___NOSYS` | `bool` (Scalari per evento) | Analysis Level | Flag di selezione per le Control Region (Z e Top e-mu) | Certo |
  | `bbtt_pass_baseline_SR___NOSYS`, `bbtt_pass_baseline_HadHad___NOSYS`, `bbtt_pass_baseline_LepHad___NOSYS`, `bbtt_pass_baseline_STT___NOSYS`, `bbtt_pass_baseline_SLT___NOSYS`, `bbtt_pass_baseline_LTT___NOSYS`, `bbtt_pass_baseline_DTT___NOSYS`, `bbtt_pass_baseline_DBT___NOSYS` | `bool` (Scalari per evento) | Analysis Level | Flag di selezione baseline per regioni e canali specifici | Certo |

  | `bbtt_pass_LepHad___NOSYS`, `bbtt_pass_LepHad_1B___NOSYS`, `bbtt_pass_LepHad_2B___NOSYS` | `bool` (Scalari per evento) | Analysis Level | Flag di selezione per il canale semileptonico (ell tau_had) | Certo |
  | `bbtt_pass_STT_1B`, `bbtt_pass_STT_2B`, `bbtt_pass_SLT_1B`, `bbtt_pass_SLT_2B`, `bbtt_pass_LTT_1B`, `bbtt_pass_LTT_2B`, `bbtt_pass_DTT_1B`, `bbtt_pass_DTT_2B`, `bbtt_pass_DBT_1B`, `bbtt_pass_DBT_2B` | `bool` (Scalari per evento) | Analysis Level | Flag di selezione per combinazioni di trigger (Single/Di-Tau, Single/Di-B-jet, ecc.) | In corso di verifica |
  | `bbtt_pass_AntiIsoLepHad___NOSYS` | `bool` (Scalare per evento) | Analysis Level | Flag per la regione di controllo con leptone non isolato usata per la stima del fondo fake | Certo |
  | `bbtt_pass_trigger_SR___NOSYS`, `bbtt_pass_trigger_DTT___NOSYS`, `bbtt_pass_trigger_STT___NOSYS`, `bbtt_pass_trigger_SLT___NOSYS`, `bbtt_pass_trigger_LTT___NOSYS`, `bbtt_pass_trigger_DBT___NOSYS` | `bool` (Scalari per evento) | Analysis Level | Decisione combinata di trigger applicata a ciascuna regione | Certo |
-->
* `[truth_]`: Verità a livello di evento/generatore (Higgs, HH, PDF)
    * **Livello**: Truth Level
    * **Stato**: Certo [Classificato da AI, in sospeso]

* `[truthjet_antikt4_]`: Jet di verità (particle-level, no detector sim)

  | Nome variabile | Formato | Livello | Descrizione | Stato |
  |---|---|---|---|---|
  | `truthjet_antikt4_pt`, `truthjet_antikt4_eta`, `truthjet_antikt4_phi`, `truthjet_antikt4_m` | `std::vector<float>` (Jagged) | Truth Level | Cinematica e massa del jet a livello di particella | Certo |
  | **`truthjet_antikt4_HadronConeExclTruthLabelID`**, `truthjet_antikt4_PartonTruthLabelID` | `std::vector<int>` (Jagged) | Truth Level | Identificatore del flavour truth basato sugli adroni nel cono o sul partone iniziale | Certo |

* `[truthtau_]`: Tau di verità

  | Nome variabile | Formato | Livello | Descrizione | Stato |
  |---|---|---|---|---|
  | **`truthtau_px`, `truthtau_py`, `truthtau_pz`, `truthtau_e`, `truthtau_m`** | `std::vector<float>` (Jagged) | Truth Level | Quadrimpulso e massa del tau a livello truth | Certo |
  | **`truthtau_IsHadronicTau`** | `std::vector<bool>` (Jagged) | Truth Level | Flag che identifica se il tau truth decade adronicamente | Certo |
  | `truthtau_classifierParticleType`, `truthtau_classifierParticleOrigin` | `std::vector<int>` (Jagged) | Truth Level | Classificazione del tipo di particella e della sua origine a livello generatore | Certo |

* `[truthelectron_]`: Elettroni di verità

  | Nome variabile | Formato | Livello | Descrizione | Stato |
  |---|---|---|---|---|
  | `truthelectron_px`, `truthelectron_py`, `truthelectron_pz`, `truthelectron_e`, `truthelectron_m`, `truthelectron_pdgId` | `std::vector<float>`, `std::vector<int>` (Jagged) | Truth Level | Quadrimpulso, massa e codice PDG dell'elettrone a livello truth | Certo |

* `[truthmuon_]`: Muoni di verità

  | Nome variabile | Formato | Livello | Descrizione | Stato |
  |---|---|---|---|---|
  | `truthmuon_px`, `truthmuon_py`, `truthmuon_pz`, `truthmuon_e`, `truthmuon_m`, `truthmuon_pdgId` | `std::vector<float>`, `std::vector<int>` (Jagged) | Truth Level | Quadrimpulso, massa e codice PDG del muone a livello truth | Certo |

* `[truthmet_]`: MET di verità

  | Nome variabile | Formato | Livello | Descrizione | Stato |
  |---|---|---|---|---|
  | `truthmet_mpx`, `truthmet_mpy`, `truthmet_sumet` | `float` (Scalare) | Truth Level | Componenti dell'impulso mancante trasverso (px, py) e energia trasversa totale sumET a livello truth | Certo |

---
## Mer 2026/09/16

### Attività
**Obiettivo 2.2**
- Indagine su BCE loss: diagnostica per-sample su validation test e train set (*è una buona misura della qualità del training per il dataset `MC23G`?*)

**Obiettivo 3.1**
- Studi preliminari sulla struttura dei file `.root` associati a $HH\to bb\tau\tau$: interpretazione del significato delle labels e confronto/analisi di coerenza tra i branches

**Altro**
- Partecipazione group meeting ATLAS Genova


### Risultati

**OSSERVAZIONE - La loss di train media e di validazione media è dominata da una coda di jet anomali, non da un fallimento generalizzato dei modelli**

Si è implementata una analisi diagnostica sulla loss di validazione e di training, ottenendo la distribuzione della loss calcolata **per-sample** (per jet) al posto che aggregata (il valore di loss sulle epoche mostrato nelle precedenti analisi era sempre la **loss media** su tutti i samples ad una epoca fissata).

*Motivazione*: approfondire la motivazione alla base di una apparente contraddizione tra due osservazioni:
- sostanziale indipendenza della best loss sul validation set dalle scelte architetturali del modello, che si assesta attorno a valori non bassi, scendendo molto poco (o per nulla) durante il training (~0.5)
- l'osservazione che la capacità di generalizzazione di questi modelli comunque non fosse pessima nonostante l'impossibilità di far scendere la loss (si vedano tabelle comparative, con AUC sul test set nell'ordine del ~0.85 in molti casi) 

Effettuando il calcolo della loss per sample sul validation set di `MC23G` (trainset completo, `~9e6` jet), utilizzando il risultato della [run](https://github.com/giumont/flavour_tagging/tree/main/outputs/MC23G_dataset/DNN/full_reduced/3layers_256_128_64_dropout0.3_BatchNorm1d_batch65536) per:
       - **Hidden layers**: `(64, 32, 16)`
        - **Dimensiona batch**: `16384`
        - **Dropout**: `0.3`
(architettura @best per trainset da 1e6 jet), si trova che **sia sul validation set che sul train set i valori di loss mostrati dagli output delle run sono dominati da outlier**: la coda delle distribuzioni `loss(sample)` è pesante in entrambi i casi:

- Validation set
Molta differenza tra il valore medio e il valore mediano della loss sui sample:

        Mean loss (deve combaciare col best_val_loss salvato): 0.48933816
        Median loss: 0.26862738
        Percentile 50: 0.2686
        Percentile 90: 1.2890
        Percentile 95: 1.6411
        Percentile 99: 2.3705
        Percentile 99.9: 3.8304
        Percentile 99.99: 4.4055
        Max loss: 5.582295
        AUC: 0.847201686455195

    ![](https://codimd.web.cern.ch/uploads/upload_6048d79a7d24f6f9f56c91aff606239e.png)
    ![](https://codimd.web.cern.ch/uploads/upload_d5a81f5ce271f5fbd5284efcd26f2380.png)
    D'altra parte, come già osservato per il test set (si vedano precedenti tabelle comparative e output su repo), la AUC presenta dei valori accettabili dopo il training, indicando che una capacità di generalizzazione è comunque presente:
    ![](https://codimd.web.cern.ch/uploads/upload_02d1a6373fbcc1fbbca056e10dcd289c.png)



- Train set 
Simile comportamento:

        Mean loss (deve combaciare col best_train_loss salvato): 0.3702514
        Median loss: 0.1832912
        Percentile 50: 0.1833
        Percentile 90: 1.0522
        Percentile 95: 1.5127
        Percentile 99: 2.1988
        Percentile 99.9: 3.2574
        Percentile 99.99: 4.6543
        Max loss: 8.573427
        AUC: 0.9114659415821524

    ![](https://codimd.web.cern.ch/uploads/upload_b7f3ac30f9b5719f8a3d7470154f6ef7.png)
    

La distribuzione è più skewed a destra per il validation set rispetto al train set:
```python
thresholds = [0.5, 1.0, 2.0, 3.0, 5.0]
print(f"{'soglia':>8} | {'% train > soglia':>18} | {'% val > soglia':>16}")
for t in thresholds:
    frac_train = (per_sample_loss_train > t).mean() * 100
    frac_val   = (per_sample_loss_val > t).mean() * 100
    print(f"{t:8.1f} | {frac_train:18.4f} | {frac_val:16.4f}")
```
```bash
   | soglia | % train | % val | rapporto val/train |
    |:------:|:-------:|:-----:|:-------------------:|
    | 0.5    | 21.60   | 31.13 | 1.44× |
    | 1.0    | 10.69   | 14.76 | 1.38× |
    | 2.0    | 1.68    | 1.89  | 1.13× |
    | 3.0    | 0.150   | 0.422 | 2.81× |
    | 5.0    | 0.0050  | 0.0006| 0.12× (train > val) |

```
In particolare, la differenza tra la media e la mediana (skewness) per queste loss è:
    - 0.489 - 0.269 = 0.220 sul validation set
    - 0.370 - 0.183 = 0.187 sul train set
quindi effettivamente in questo esempio l'uso della media per le loss su sample sottostima (leggermente) la performance sul validation set relativamente al train set, rispetto ad utilizzare la mediana.


**Commenti**

- La BCE loss **non è una buona metrica assoluta per confrontare le run** su questo dataset: la sua media è dominata da una coda pesante di sample con loss anomala, sia su train che su validation. Usare la mediana (o l'AUC, che isola la componente di vera capacità discriminante ) come criterio di confronto/early stopping restituirebbe un quadro più stabile e meno sensibile a quanti jet-outlier capitano in un particolare batch/set.

- **Non tutto il gap train-val è però un artefatto della metrica**: calcolando AUC con lo stesso identico metodo su entrambi i set (train 0.9115, val 0.8472), il calo è reale e non trascurabile (Δ AUC ≈ 0.064, corrispondente a una riduzione di circa il 15-16% della capacità discriminante "sopra il caso" — (0.9115-0.5) → (0.8472-0.5)). Quindi le considerazioni sopra sulla metrica non basterebbero a risolvere totalmente l'overfitting osservato.

- **La skewness (differenza media−mediana) è simile in ordine di grandezza tra i due set**: 0.220 sul validation, 0.187 sul train. Questo suggerisce che una parte sostanziale della coda pesante è una **proprietà intrinseca del dataset/della loss stessa**, presente e comparabile in entrambi gli split, non un fenomeno esclusivo della validazione 

    -> **si rafforza l'ipotesi che sia necessario intervenire sul dataset per migliorare significativamente le performance, non solo sull'architettura** (si potrebbe ad esempio correlare la posizione nella coda per la loss con le feature cinematiche dei jet)



**PRELIMINARE - Qual è la frazione di oggetti ricostruiti che sono effettivamente selezionati per l'analisi, nei file `.root` su $HH\to bb\tau\tau$?**

Considerando i branch del `TTree`:
    - `recojet_antikt4PFlow_*`: jet ricostruiti
    - `tau_*`: tau ricostruiti
vogliamo determinare se le relative collezioni sono già pre-filtrate a livello analisi, tramite l'applicazione delle flag:
    - `recojet_antikt4PFlow_eta[recojet_antikt4PFlow_isAnalysisJet___NOSYS]`
    - `tau_eta[tau_isAnalysisTau___NOSYS]`
dove `*_eta` è una variabile cinematica (type `std::vector<float>`, in formato di *array jagged*), che quindi si presuppone associata ad ogni singolo evento "raw" ricostruito. 

Considerando tutti i file `.root` a disposizione, si ottengono le seguenti statistiche cumulative:
```
----------------------------------------------------------------------------------------------------
jet: recojet_antikt4PFlow_isAnalysisJet___NOSYS
----------------------------------------------------------------------------------------------------

  STATISTICA AGGREGATA SU TUTTI I FILE
      eventi totali usati:           275946
      oggetti totali:               1570713
      oggetti con flag=True:        1563504
      frazione True:             0.995410 (99.541%)
      media oggetti/evento:      5.692103
      media True/evento:         5.665978

----------------------------------------------------------------------------------------------------
tau: tau_isAnalysisTau___NOSYS
----------------------------------------------------------------------------------------------------

  STATISTICA AGGREGATA SU TUTTI I FILE
      eventi totali usati:           275946
      oggetti totali:                564238
      oggetti con flag=True:         564238
      frazione True:             1.000000 (100.000%)
      media oggetti/evento:      2.044741
      media True/evento:         2.044741
```

dunque gli eventi sono pre-filtrati nel caso dei tau, e con filtro a monte trascurabile nel caso dei jet. 
**Nel seguito, si applicano comunque le due maschere `isAnalysys*`, ma questa osservazione esclude di ripetere le analisi fatto per il caso "raw" non filtrato, vista la piccola o nulla deviazione tra l'applicazione o meno delle flag**. 

*NOTA*: le deviazioni per singolo file di queste percentuali sono molto piccole (Delta sulle frazioni di True $\sim 10^{-3}-10^{-4}$).

Per quanto riguarda la media oggetti/evento: 5.69 jet/evento e 2.04 tau/evento. Il valore ottenuto per i $\tau$ è coerente con il decadimento $H\to \tau\tau$, mentre **il numero di jet per evento sembra sovrastimato** (da approfondire).


**PRELIMINARE - Distribuzione delle truth label sugli eventi di jet e tau, nei file `.root` su $HH\to bb\tau\tau$**

Consideriamo la distribuzione delle label di verità associate ai branch del `TTree`:
    - `recojet_antikt4PFlow_*`: jet ricostruiti
    - `tau_*`: tau ricostruiti
per intepretarne il significato e avere informazioni preliminari sul tipo di eventi simulati, a truth-level.

- Per i jet ricostruiti:
    - ` recojet_antikt4PFlow_HadronConeExclTruthLabelID`: truth flavour associato al jet
    
        ``` ----------------------------------------------------------------------------------------------------
        recojet_antikt4PFlow_HadronConeExclTruthLabelID
        ----------------------------------------------------------------------------------------------------

            SELEZIONE ATTIVA: recojet_antikt4PFlow_isAnalysisJet___NOSYS

            Distribuzione AGGREGATA:
                         0  count=    552300  ( 35.32%)
                         5  count=    499121  ( 31.92%)
                        15  count=    481078  ( 30.77%)
                         4  count=     31005  (  1.98%)

        ```
     Da [documentazione ufficiale ATLAS su analysis OpenData](https://atlas-physlite-content-opendata.web.cern.ch/), la variabile `HadronConeExclTruthLabelID` indica, nel contesto dei jet di analisi (`AnalysisJets`):
    >   	Truth labels (5=B, 4=C, 15=tau), used by flavour tagging group

    Quindi **la percentuale del ~30% per jet ricostruiti associati, a livello di verità, a $\tau$ adronici è una prima misura quantitativa dell'overlap tra queste due categorie, prima di qualsiasi overlap removal**.


- Per i $\tau$ ricostruiti:
    - `tau_truth_IsHadronicTau`: flag truth sul tau adronico
    
        ```  ----------------------------------------------------------------------------------------------------
        tau_truth_IsHadronicTau
        ----------------------------------------------------------------------------------------------------

            SELEZIONE ATTIVA: tau_isAnalysisTau___NOSYS

            Distribuzione AGGREGATA:
                         1  count=    411283  ( 72.89%)
                         0  count=    152955  ( 27.11%)
        ```
        
        **La percentuale di tau non adronici NON è trascurabile: bisogna applicare questa flag nelle prossime analisi**
    
    - `tau_truthType`: 

        ```
        ----------------------------------------------------------------------------------------------------
        tau_truthType
        ----------------------------------------------------------------------------------------------------

            SELEZIONE ATTIVA: tau_isAnalysisTau___NOSYS

            Distribuzione AGGREGATA:
                         1  count=    411283  ( 72.89%)
                         5  count=    104363  ( 18.50%)
                         4  count=     20977  (  3.72%)
                         3  count=     17389  (  3.08%)
                         0  count=      8923  (  1.58%)
                         2  count=      1303  (  0.23%)
        ```                 
        **La percentuale di `tau_truthType=1` coincide esattamente con quella di `tau_truth_isHadronicTau=True`: per l'analisi da ora in poi si utilizza direttamente `tau_truth_isHadronicTau` in quanto più semanticamente chiara**.
    - `tau_truthOrigin`:
    
        ```   ----------------------------------------------------------------------------------------------------
            tau_truthOrigin
            ----------------------------------------------------------------------------------------------------

            SELEZIONE ATTIVA: tau_isAnalysisTau___NOSYS

            Distribuzione AGGREGATA:
                     0  count=    564238  (100.00%)
        ```
        
        **Non popolata: non utilizzata nel seguito**
        
    - `tau_tauTruthJetLabel`: [INTERPRETAZIONE PLAUSIBILE- DA CHIARIRE] potrebbe indicare il flavour di verità del jet più vicino/sovrapposto

        ```
        ----------------------------------------------------------------------------------------------------
        tau_tauTruthJetLabel
        ----------------------------------------------------------------------------------------------------

            SELEZIONE ATTIVA: tau_isAnalysisTau___NOSYS

            Distribuzione AGGREGATA:
                        -1  count=    346919  ( 61.48%)
                         5  count=     94715  ( 16.79%)
                        21  count=     90465  ( 16.03%)
                       -99  count=      9450  (  1.67%)
                         2  count=      8651  (  1.53%)
                         1  count=      6643  (  1.18%)
                         3  count=      4141  (  0.73%)
                         4  count=      3254  (  0.58%)
        ```
        
        Se questa fosse la corretta interpretazione, e interpretando gli indici con le convenzioni normalmente utilizzate su `PartonID`, allora si avrebbe:
        ```
        -1   61.48%   nessun match/label valido
         5   16.79%   b
        21   16.03%   gluone
        -99   1.67%   sentinella
         2    1.53%   u
         1    1.18%   d
         3    0.73%   s
         4    0.58%   c
         ```
         
         **La percentuale di associazione di vicinanza dei tau ad un b-jet di verità potrebbe essere un'altra info utile, ma bisogna prima confermare il significato delle label e del parametro stesso** 
         
         
**PRELIMINARE - Associazione di verità di jets e $\tau$ con Higgs, nei file `.root` su $HH\to bb\tau\tau$**

Si considerando le bitmasks `*_parentHiggsParentMask`, interpetate come:
```
    valore=     1  binario=000000000001  "Figlio di H1"
    valore=     2  binario=000000000010  "Figlio di H2"
    valore=     0  binario=000000000000  "Non figlio di un H"
    valore=     3  binario=000000000011  "Associato contemporaneamente a entrambi gli H"
```

- Tau:
    - `tau_parentHiggsParentsMask`:


    ``` ----------------------------------------------------------------------------------------------------
            tau_parentHiggsParentsMask
            ----------------------------------------------------------------------------------------------------
                SELEZIONE ATTIVA: tau_isAnalysisTau___NOSYS
                Valori AGGREGATI e bit attivi (valore | binario | count | frazione)
                  valore=     1  binario=000000000001  count=    211779  ( 37.53%)
                  valore=     2  binario=000000000010  count=    210939  ( 37.38%)
                  valore=     0  binario=000000000000  count=    141363  ( 25.05%)
                  valore=     3  binario=000000000011  count=       157  (  0.03%)

                SELEZIONE ATTIVA: tau_isAnalysisTau___NOSYS + tau_truth_IsHadronicTau
                Valori AGGREGATI e bit attivi (valore | binario | count | frazione)
                  valore=     1  binario=000000000001  count=    203501  ( 49.48%)
                  valore=     2  binario=000000000010  count=    202698  ( 49.28%)
                  valore=     0  binario=000000000000  count=      4956  (  1.21%)
                  valore=     3  binario=000000000011  count=       128  (  0.03%)
    ```
 
    **Da notare come i tau adronici di verità sono associati praticamente sempre ad un H**
 
- Jets: 
    - `recojet_antikt4PFlow_parentHiggsParentsMask`

    ```
    ----------------------------------------------------------------------------------------------------
    recojet_antikt4PFlow_parentHiggsParentsMask
    ----------------------------------------------------------------------------------------------------

        SELEZIONE ATTIVA: recojet_antikt4PFlow_isAnalysisJet___NOSYS

        Valori AGGREGATI e bit attivi (valore | binario | count | frazione)
          valore=     0  binario=000000000000  count=    984845  ( 62.99%)
          valore=     2  binario=000000000010  count=    289482  ( 18.51%)
          valore=     1  binario=000000000001  count=    288811  ( 18.47%)
          valore=     3  binario=000000000011  count=       366  (  0.02%)
    ```
    
    **Da notare come invece per i jets questo non sia assolutamente detto, come atteso**
    
    - Cross check: qual è il rapporto truth-level, per un jet, tra avere flavour b e essere figlio di un H?

    ```
    ----------------------------------------------------------------------------------------------------
    Cross-check jet: HadronConeExclTruthLabelID == 5 vs parentHiggsParentsMask
    ----------------------------------------------------------------------------------------------------
        SELEZIONE ATTIVA: recojet_antikt4PFlow_isAnalysisJet___NOSYS

      STATISTICA AGGREGATA
          n. jet totali:                            1563504
          n. jet con label==5 :                 499121  (31.923%)
          n. jet con mask!=0:                       578659  (37.010%)
          n. jet con label E mask!=0:               483296  (30.911%)
          n. jet con label E mask==0:                15825  ( 1.012%)

          P(mask!=0 | label==5):            0.968294  (96.829%)
          P(label==5 | mask!=0):            0.835200  (83.520%)

          Distribuzione di parentHiggsParentsMask tra i jet con label==5:
            mask=0  binario=000000000000 count=     15825 ( 3.171%)
            mask=1  binario=000000000001 count=    241411 (48.367%)
            mask=2  binario=000000000010 count=    241542 (48.393%)
            mask=3  binario=000000000011 count=       343 ( 0.069%)

          n. jet label==5 con mask!=0:       483296 (96.829%)
          n. jet label==5 con mask==0:        15825 ( 3.171%)
          n. jet label==5 con mask==3:          343 ( 0.069%)

          CHECK:
            P(mask!=0 | label==5) = 0.968294 (96.829%)
            P(label==5 | mask!=0) = 0.835200 (83.520%)
    ```
    
    - Cross check: qual è il rapporto truth-level, per un jet, tra derivare da un $\tau$ adronico e essere figlio di un H?

    ```
    ----------------------------------------------------------------------------------------------------
    Cross-check jet: HadronConeExclTruthLabelID == 15 vs parentHiggsParentsMask
    ----------------------------------------------------------------------------------------------------
    SELEZIONE ATTIVA: recojet_antikt4PFlow_isAnalysisJet___NOSYS

      STATISTICA AGGREGATA
          n. jet totali con label==15:       481078

          Distribuzione di parentHiggsParentsMask tra i jet con label==15:
            mask=0  binario=000000000000 count=    387941 (80.640%)
            mask=1  binario=000000000001 count=     46286 ( 9.621%)
            mask=2  binario=000000000010 count=     46830 ( 9.734%)
            mask=3  binario=000000000011 count=        21 ( 0.004%)

          n. jet label==15 con mask!=0:        93137 (19.360%)
          n. jet label==15 con mask==0:       387941 (80.640%)
          n. jet label==15 con mask==3:           21 ( 0.004%)

          CHECK:
            P(mask!=0 | label==15) = 0.193601 (19.360%)
    ```
    
     **Commenti**
        - A prescindere dal flavour, si ottiene una distribuzione simmetrica tra le associazioni ad H1 e quelle a H2: questa è una prova empirica del fatto che **i due H NON sono indicizzati sulla base di come decadono** (quindi NON è $H_1\to bb$ e $H_2\to \tau\tau$ necessariamente), ma probabilmente in ordine di comparsa


**OSSERVAZIONE - La maggioranza dei jets con `HadronConeExcTruthID==15` non sono associati ad un $H$, anche se i $\tau$ dello stesso evento sono associati ad un $H$ (come atteso)**: 

[BOZZA]
Si è notato che, per la maschera `parentHiggsParentsMask`:
- Per i $\tau$ adronici, la percentuale di provenienza da un $H$ è molto alta come atteso, con solo:
`
                  valore=     0  binario=000000000000  count=      4956  (  1.21%)
`

- Ci si aspettavano quindi dei valori comparabili per i jet con etichetta di verità `HadronConeExcTruthID == 15`; per questi invece si ottiene
`
          n. jet label==15 con mask==0:       387941 (80.640%)
`

Quindi sembrano esserci molti eventi in cui 

Inoltre, il numero totale di eventi associati dopo il flatten è molto diverso. 
[FINE BOZZA]

Effettuiamo un test per capire se, **per lo stesso evento**, le associazioni del $\tau$ adronico **con un H** (`parentsHiggsParentsMask!=0` per i $\tau$) e quello dei tau-jet (che dovrebbero essere corrispondenti) concidono:

- consideriamo prima unicamente i tau-jets che sono associati effettivamente ad un $H$ (`parentsHiggsParentsMask!=0` per i jet):
    ```
    ====================================================================================================
    STEP 1 - Coerenza mask jet(label==15, mask!=0) vs tau veri dello stesso evento
    ====================================================================================================
       jet label==15 totali: 481078
       di cui con mask==0 (baseline esclusa): 387941  (80.640%)
       jet label==15 con mask!=0: 93137  (19.360%)

       jet label==15 & mask!=0 confrontati in eventi validi: 85581  (91.887% dei label==15 & mask!=0)

       jet confrontati: 85581
       coerenti (stessa mask del tau vero): 85437  (99.832%)
       INCOERENTI (mask diversa): 144  (0.168%)

    ====================================================================================================
    STEP 2 - DeltaR(jet, tau vero piu' vicino) per coerenti vs incoerenti
    ====================================================================================================
       SUBSET USATO:
          HadronConeExclTruthLabelID == 15
          parentHiggsParentsMask != 0
          stesso identico subset dello STEP 1

       DeltaR min - jet COERENTI con la mask del tau: n=     85437  mean=0.0986  median=0.0091  p10=0.0028  p90=0.0365
       DeltaR min - jet INCOERENTI con la mask del tau: n=       144  mean=0.7028  median=0.0241  p10=0.0051  p90=2.8766
    ```

*NOTA*: qui per "eventi validi" si ottengono gli eventi tali per cui:
    - esiste almeno un $\tau$ con `parentsHiggsParentsMask!=0`
    - se c'è più di un $\tau$, sono associati allo stesso Higgs (stessa `parentsHiggsParentsMask`), come ci si aspetterebbe dalla fisica

- Se però viceversa consideriamo gli eventi tau-jets "sospetti", ovvero quelli con `parentsHiggsParentsMask==0`: 
    ```
    ====================================================================================================
    STEP 1 - Coerenza mask jet(label==15, mask==0) vs tau veri dello stesso evento
    ====================================================================================================
       jet label==15 totali: 481078
       di cui con mask==0: 387941  (80.640%)

       jet label==15 & mask==0 confrontati in eventi validi: 369549  (95.259% dei label==15 & mask==0)

       jet confrontati: 369549
       coerenti (stessa mask del tau vero): 0  (0.000%)
       INCOERENTI (mask diversa): 369549  (100.000%)

    ====================================================================================================
    STEP 2 - DeltaR(jet, tau vero piu' vicino) per coerenti vs incoerenti
    ====================================================================================================
       SUBSET USATO:
          HadronConeExclTruthLabelID == 15
          parentHiggsParentsMask == 0
          stesso identico subset dello STEP 1

       DeltaR min - jet COERENTI: nessun valore disponibile
       DeltaR min - jet INCOERENTI con la mask del tau: n=    369549  mean=0.2735  median=0.0143  p10=0.0045  p90=1.3841
    ```

    *NOTA*: il fatto che la percentuale di coerenza sia zero qui è vero per costruzione (stiamo prendendo solo i $\tau$ con `parentsHiggsParentsMask!=0`)

**Commenti**:
- Il tasso di coerenza su `tau_` è molto alto per l'associazione agli Higgs: due $\tau$ nello stesso evento risultano associati allo stesso Higgs il $\sim 99\%$ delle volte ("eventi validi") per il caso `jet(label==15, mask!=0)`
- Considerando solo gli eventi con `parentsHiggsParentsMask!=0` per i jet, il tasso di coerenza tra l'higgs a cui è associato il tau-jet e quello a cui è associato il corrispondente $\tau$ è molto alto ($\sim 99\%$)
- Inoltre, imponendo questa maschera, i casi di incoerenza si distinguono per un $\Delta R_{min}$ del jet molto più alto rispetto ai casi coerenti: coni più larghi sono spesso associati a ....[COMPLETA]
- D'altra parte, anche gli eventi "sospetti" (con `parentsHiggsParentsMask==0` per i jet ma invece $\tau$ associato ad un higgs) presentano $\Delta R_{min}$ più alti rispetto al caso coerente di sopra. 

-> imporre la maschera  `parentsHiggsParentsMask!=0` sui $\tau$-jet a priori garantisce alta coerenza, anche se **rimane non chiaro il criterio di associazione di questa maschera ai tau-jet**; sembra che l'informazione portata da `parentHiggsParentsMask` sul jet non si propaga in modo affidabile per la sottopopolazione `label==15`, ma le cause profonde di questo comportamento restano **non identificate** dal solo confronto delle mask

<!--
Guarda questi due numeri, presi da controlli diversi:

Jet con label==15 e parentHiggsParentsMask≠0 (jet "vicino a un tau" e effettivamente figlio di un Higgs): 93137
Tau con tauTruthJetLabel==5 (tau "vicino a un jet vero di flavour b"): 94715

Sono quasi identici (differenza ~1.7%). Questo non è necessariamente un caso: sono candidati naturali a rappresentare la stessa popolazione fisica vista da due lati — cioè le coppie genuine in cui un vero b-jet (da H→bb) e un vero tau adronico (da H→ττ) sono effettivamente vicini in ΔR nello stesso evento, presumibilmente nel regime boosted dove pT(HH) alto comprime la topologia (esattamente il tipo di correlazione ipotizzata nei tuoi appunti iniziali, "come cambia ΔR(τ,b) al variare del regime energetico").
-->


[BOZZA] L'intepretazione "mask Higgs sul jet = 0 quando il tau non viene ricostrito separamente dal jet" non sembra reggere:
`
 jet orfani (nessun tau nello stesso evento): 49373  (3.158% dei jet totali)
 <<<
 n. jet label==15 con mask==0:       387941 (80.640% dei jet con label==15)
` 


---
## Gio 2026/09/17

### Attività

**Obiettivo 3.1**
- Studi preliminari sulla struttura dei file `.root` associati a $HH\to bb\tau\tau$: interpretazione del significato delle labels e confronto/analisi di coerenza tra i branches
- Libreria [`Akward Array`](https://awkward-array.org/doc/main/index.html) Python: studio sintassi metodi
    - `ak.sum`
    - `ak.num`
    - `ak.cartesian`
    - `ak.unzip`
- Studio del $\tau$ tagger (parametro `tau_GNTauScoreSigTrans_v0prune`): mapping efficienza di segnale $\epsilon_\tau$ $\leftrightarrow$ score continuo $\in [0,1]$ dato dalla variabile (in particolare, determinazione del working point `TAU_SCORE_WP85_THRESHOLD`)
- Studio overlap geometrico (tramite $\Delta R : = \sqrt{(\Delta \eta)^2 + (\Delta \phi)^2}$) e **conseguente scelta di una soglie di overlap** per $\tau$ e b-jets ricostruiti (subset con `isAnalysisLevel == 1`) con:
    - $\tau$ ricostruiti: dati da `tau[tau_GNTauScoreSigTrans_v0prune >= TAU_SCORE_WP85_THRESHOLD]`
    - $b$-jet ricostruiti: dati da 
`recojet[recojet_antikt4PFlow_ftag_select_GN2v01_FixedCutBEff_85==1]`
Corrispondenti categorie di verità identificate da:
    - $\tau$ di verità: `tau_isHadronicTau == 1`
    - $b$-jet di verità: `recojet_antikt4PFlow_HadronConeExclTruthLabelID == 5`

    In particolare:
    - istogramma $\Delta R(\tau, b$-jet) (pair-level) al variare della combinatoria: valore minimo per ogni b-jet ricostruito / valore minimo per ogni $\tau$ ricostruito / valori su tutte le coppie ($\tau, b$-jet) ricostruite per evento
    - istogramma $\Delta R(\tau, b$-jet) (pair-level) al variare della categoria di verità (per tutte le coppie ricostruite a livello di evento):
        - tau vero, b-jet vero
        - tau vero, b-jet fake
        - tau fake, b-jet vero
        - tau fake, b-jet fake

**Altro**
- Partecipazione group meeting ATLAS Genova


### Risultati

**PRELIMINARE - Determinazione dei working point associati al tagger $\tau$ `tau_GNTauScoreSigTrans_v0prune`**:
Partendo dal metodo `sklearn.metrics.roc_curve`, che returna `tpr` (efficienza segnale) e `fpr` (efficienza fondo) al variare dello score e delle reali label, determinati working points (score corrispondenti a efficienze di segnale di interesse):

```
====================================================================================================
SOGLIE OPERATIVE COMPARABILI AL B-TAGGING - famiglia di working point [70, 75, 80, 85, 90, 97]%
====================================================================================================
    target eff.   soglia score  eff. misurata         bkg eff.   bkg rejection
            70%       0.324070        70.001%           1.243%           80.46
            75%       0.270537        75.001%           1.631%           61.30
            80%       0.217226        80.003%           2.175%           45.97
            85%       0.163094        85.000%           2.980%           33.56
            90%       0.108341        90.003%           4.346%           23.01
            97%       0.027386        97.000%          10.809%            9.25
```

![](https://codimd.web.cern.ch/uploads/upload_f8d63b4b8f2a51ef2b8c71bf551bd7f3.png)

<!--
Nel seguito, lo score corrispondente ad una **efficienza dell'$85\%$** verrà usato per diretta comparazione con la maschera booleana del b-tagger `   "recojet_antikt4PFlow_ftag_select_GN2v01_FixedCutBEff_85"` già data come variabile nel dataset. 
-->

**RISULTATO - Scelta della soglia geometrica $\Delta R$ di overlap tra $\tau$ e jet ricostruiti** 
[AGGIUNGI]

Si vuole studiare l'overlap geometrico (distribuzione di $\Delta R$) tra i $\tau$ e i $b$-jet identificati come tali dai corrispondenti tagger nello stesso evento. 

- Per confronto, si plottano tre conteggi differenti per $\Delta R$ per ogni evento: valore minimo per ogni b-jet ricostruito / valore minimo per ogni $\tau$ ricostruito / valori su tutte le coppie ($\tau, b$-jet) ricostruite per evento:
![](https://codimd.web.cern.ch/uploads/upload_97f651d4275c6b9a703ab532c2fcf954.png)

<!-- VERSIONE VECCHIA DOVE FISSAVO A PRIORI EFFICIENZA DELL 85% SUI DUE TAGGER
    ![](https://codimd.web.cern.ch/uploads/upload_e2d1cb15a51cfc71c4d2a5e68aff6894.png)
Le due soglie di $\Delta R$ (linee rosse) plottate sono due valori usati per identificare l'overlap trovate in letteratura, da considerare come riferimento [AGGIUNGI FONTI]. 

    Nell'analisi si sono inoltre scartati gli eventi "orfani", ovvero quelli in cui non era presente nessun b-jet ricostruito o nessun $\tau$ ricostruito:
    [AGGIUNDI OUTPUT CONTEGGI]
    ```
    eventi totali: 275946
   b-jet selezionati totali: 464594
   tau selezionati totali: 354148

   jet orfani (nessun tau nello stesso evento): 68122  (14.663% dei jet totali)
   tau orfani (nessun jet nello stesso evento): 21473  (6.063% dei tau totali)
   ```
-->
 
Nell'analisi si sono inoltre scartati gli eventi "orfani", ovvero quelli in cui non era presente nessun b-jet ricostruito o nessun $\tau$ ricostruito:
    [AGGIUNDI OUTPUT CONTEGGI]
    ```
    ```
    Per $\Delta R$ piccoli, i conteggi associati alle tre misure di $\Delta R$ sono sovrapposti: **ogni oggetto reco ha esattamente un solo partner vicino**.
    **NOTA**: questa osservazione NON garantisce che i due oggetti siano in realtà lo stesso a livello truth, potrei ancora avere due oggetti distinti ma fisicamente vicini, l'informazione che ci dà questa osservazione è solo che **saranno non più di due**!
    
Per le prossime analisi, scegliamo la soglia che definisce overlap geometrico 
 $$
    \boxed{\Delta R_{overlap } = 0.4}
$$
<!--
    **Scegliamo la soglia di overlap per $\Delta R$ come il massimo valore (a due cifre decimali) tale per cui la sovrapposizione dei tre istogrammi è del 100%**: otteniamo una soglia 
    $$
    \boxed{\Delta R_{overlap } = 0.33}
    $$
    ![](https://codimd.web.cern.ch/uploads/upload_88bae7f5f4f8a059650c12ca881f08f7.png)
    
    La frazione di coppie overlappanti rispetto al totale di coppie (secondo le diverse definizioni combinatoriche) è data da 
    ```
       --- soglia DeltaR < 0.3300 (SOGLIA PERFETTO OVERLAP) ---
   jet con almeno un tau sovrapposto: 9631 / 396472  (2.429% dei jet con >=1 tau nell'evento)
   tau con almeno un jet sovrapposto: 9631 / 332675  (2.895% dei tau con >=1 jet nell'evento)
   coppie (jet,tau) sovrapposte: 9631 / 576796  (1.670%)
   ```

    **NOTA**: la determinazione di una soglia esatta per $\Delta R_{overlap}$ è fortemente dipendente dal dataset. Verrà utilizzata nel seguito come taglio per studiare le caratteristiche cinematiche per l'overlap **in questo dataset**, ma usarla per determinare il trainset potrebbe peggiorare la generalizzazione della NN, per cui si troverà una strategia diversa più avanti. 

    **Commenti**
    - La molteplicità di eventi sui b-jet è maggiore di quella sui $\tau$. In precedenza si era trovata una molteplicità $\sim 2$ di $\tau$ per evento e una $\sim 5$ di jet **di qualsiasi flavour**; questa è una nuova informazione che invece riguarda solo i jet taggati come b-jet. 
    Ha senso che le code a grandi $\Delta R$ siano più pesanti per le misure su b-jet rispetto alle misure su $\tau$ perché, essendoci un eccesso di $b$-jet taggati rispetto a quelli che la fisica di $HH\to bb\tau\tau$ prevederebbe (stessa molteplicità dei $\tau$), l'eccesso deve essere costituito da jet non correlati ai $\tau$, quindi a distanze angolari
    - E' atteso che la molteplicità di tutte le coppie possibili sia maggiore per combinatoria, e anche che questa distribuzione abbia code più pesanti (prendo tutte le coppie in un evento, che possono non essere correlate).
-->

**OSSERVAZIONE - Gli overlapping di verità ($\tau$ e $b$-jet reali e con $\Delta R < 0.4$) sono pochi, sia relativamente agli overlapping di fake sia relativamente ai conteggi totali di verità: nella maggior parte dei casi l'overlapping è effettivamente errore di classificazione**

Dal grafico prodotto (pair-level: può esserci più di una coppia per evento + considero **tutte** le possibili coppie ricostruite $\tau$ + $b$-jet per evento), si vede che, **normalizzando gli istogrammi** (`density=True`):
- le classificazioni totalmente corrette (linea verde) difficilmente si trovano a $\Delta R$ molto bassi (fisicamente, è difficile trovare un vero b-jet e un vero tau così vicini)
- le categorizzazioni errate si concentrano nella regione di overlap: se abbiamo una classificazione errata, è probabile che riguardi un oggetto nella regione di overlap ("se i due tagger stanno guardando davvero lo stesso oggetto, almeno uno dei due deve star sbagliando") 
![](https://codimd.web.cern.ch/uploads/upload_5010a13be56c6affa90942fe71425e97.png)


Notiamo inoltre che gli eventi con jet taggato male sono più frequenti degli eventi con tau taggato male, ma poiché stiamo considerando tutte le combinazioni di coppie possibili (contando ogni oggetto più di una volta) e i tau sono di piu dei jet per evento, ha senso che sia così. 



<!--
**Obiettivo 3.2:** sul campione di segnale, quantificare la bontà della classificazione, prima separatamente per $b$-jet e $\tau$-jet e poi provando a combinare "manualmente" i discriminanti di identificazione in caso di overlap

- (in funzione di pT del jet score del GN2)
- derivati su ttbarra: potrebbe esserci differenza tra soglia efficienza riportata e reale
-->

---
## Ven 2026/09/18

### Attività
**Obiettivo 3.1**
- Studio delle caratteristiche cinematiche degli oggetti reco overlappati e del loro potere discriminante:
    - $p_T, \phi, \eta$ del b-jet reco per le quattro categorie di verità
    - $p_T, \phi, \eta$ del $\tau$ reco per le quattro categorie di verità

**Obiettivo 3.2**
- Studio dell'efficienza di GN2 sul dataset $HH\to bb\tau\tau$: le efficenze nominali corrispondono a quelle reali?
- Studio della correlazione tra score di GN2 e score del $\tau$-tagger in caso di overlap:
    - Produzione violin plot (categorie discrete quantili GN2 VS score continui $\tau$-tagger)

### Risultati
**PRELIMINARE - Le efficienze reali di GN2 sul dataset $HH\to bb\tau\tau$ sono in ottimo accordo con le efficienze nominali del tagger**
[DA COMPLETARE]

**RISULTATO - Lo score 
<!-- 
sostituire coeff pearson con metrica corretta
-->


---
## Sab 2026/09/19

### Attività
**Obiettivo 3.1**
- Studio delle caratteristiche degli oggetti reco overlappati e del loro potere discriminante, per le quattro categorie di verità:
    - massa del jet: `recojet_antikt4PFlow_m___NOSYS`
    - numero *muoni soft* [APPROFONDIRE] associati al jet: `recojet_antikt4PFlow_n_muons___NOSYS`
    - molteplicità di tracce del $\tau$: `tau_nProng`
    - modo di decadimento del $\tau$: `tau_decayMode`
    - carica del $\tau$: `tau_charge`

**Obiettivo 3.2**
- Produzione delle metriche di valutazione associate al b-tagger GN2 e al $\tau$-tagger GNTau separatamente sul dataset  $HH\to bb\tau\tau$:
    - ROC 
    - confusion matrix per soglia $\epsilon = 85\%$
    - efficiency vs score threshold
    - background rejection vs efficiency
    - score distribution

### Risultati
**RISULTATI - Potere discriminante di alcune caratteristiche dei jet ricostruiti**
-  Numero muoni soft `recojet_antikt4PFlow_n_muons___NOSYS`: solo i veri b-jets (T_) ne hanno un numero diverso da zero

    ![](https://codimd.web.cern.ch/uploads/upload_b8b337271eb7026eeabdfea565440141.png)
    
- Massa del jet `recojet_antikt4PFlow_m___NOSYS`: i veri b-jets (T_)
    - presentano picchi a masse maggiori ($5-10\space GeV$ VS $2-5\space GeV$ delle categorie (F_)) 
    - hanno code di massa più lunghe rispetto alle categorie (F_)

    Questi risultati sono coerenti con $m_b >> m_\tau$. Inoltre:
    - categoria (FF) ha code di massa intermedie: gli oggetti che non sono $\tau$ / non sono $b$-jets potrebbero essere light-jets, $c$-jets (massa intermedia tra b e $\tau$), gluoni (molteplicità particelle piu alte) -> combinazione questi contributi
        ![](https://codimd.web.cern.ch/uploads/upload_6506b08a317bfb24fbf89c3de56ef07c.png)



**RISULTATI - Potere discriminante di alcune caratteristiche dei $\tau$ ricostruiti**
- Modalità di decadimento del $\tau$ adronico reco `tau_decayMode` (numero di tracce + tipo di adroni nel decadimento)
    Discrimina soprattutto i (FT) dai (TF)
    [INTERPRETAZIONE DEGLI INDICI DA TROVARE]
    
    ![](https://codimd.web.cern.ch/uploads/upload_96422b663b1edf45c83a04e1fa71b5c9.png)

- Numero di prodotti di decadimento del $\tau$ adronico reco `tau_nProng`
    Discrimina soprattutto i (FT) dai (TF): **i b-jet scambiati per $\tau$ generano nell'algoritmo del $\tau$ più prodotti di decadimento, essenso composti da una molteplicità di tracce** 
    ![](https://codimd.web.cern.ch/uploads/upload_e99765c0a2b2233cf266615251f173e3.png)

**RISULTATI - Potere discriminante del momento trasverso $p_T$ per tau e jet ricostruiti**
La distribuzione dei $p_T$ degli oggetti reco per la categoria (FF) risulta più piccata attorno a valori bassi ($\sim 10\space GeV$)
- Jet ricostruiti:
![](https://codimd.web.cern.ch/uploads/upload_d18d35b906133c66b6f319ec6f236a56.png)

- $\tau$ ricostruiti:
![](https://codimd.web.cern.ch/uploads/upload_7186f9cad2abc2444fae7a36d8a6bcc6.png)


**OSSERVAZIONE - La variabile del tagger $\tau$ `tau_GNTauScoreSigTrans_v0prune` è *signal transformed***

Plottando la distribuzione degli score su segnale (`tau_isHadronicTau == 1`) e fondo (`tau_isHadronicTau == 0`) si nota che la **distribuzione di segnale è uniforme lungo l'asse degli score**
![](https://codimd.web.cern.ch/uploads/upload_3c70e514674468987d7000b293075a32.png)

Si tratta di un comportamento atteso: come dice il nome della variabile, essa è *signal transformed* (distribuzione di segnale uniforme sull'intervallo $[0,1]$ degli score); in questo modo, **applicare una soglia $x$ seleziona una frazione del segnale pari a $1-x$ ($\epsilon_\tau = 1-x$)**. 

Questo è confermato e coerente con il calcolo dei working points ottenute nell'analisi, ad esempio:
- applicare la soglia corrispondente 0.163094 corrisponde a $\epsilon_\tau = 85%$ perché infatti $1-0.163\approx 0.837$
```
    target eff.   soglia score  eff. misurata         bkg eff.   bkg rejection
            85%       0.163094        85.000%           2.980%           33.56
```
- applicare la soglia corrispondente 0.324070  corrisponde a $\epsilon_\tau = 70%$ perché infatti $$1 - 0.324 \approx 0.676$
```
    target eff.   soglia score  eff. misurata         bkg eff.   bkg rejection
            70%       0.324070        70.001%           1.243%           80.46
```


---
## Dom 2026/09/20

### Attività
**Obiettivo 3.1**
- Studio delle caratteristiche degli oggetti reco overlappati e del loro potere discriminante, per le quattro categorie di verità:
    - MET proiettato sulla direzione del $\tau$ reco: $MET_{\parallel, \tau}$ (valore ricavato)
    - massa trasversa $\tau$ reco - MET: $m_T$ (valore ricavato)
    - score del tagger $\tau$ GNTau: `tau_GNTauScoreSigTrans_v0prune`
    - quantile del tagger $b$-jets GN2:
    `recojet_antikt4PFlow_ftag_quantile_GN2v01_Continuous`
    
### Risultati
**RISULTATO - Performance dei tagger $\tau$ GNTau e b-jet GN2 al variare della categoria di verità**
        Per la categoria (TT), entrambi i tagger ($\tau$ e b-jet) tendono ad associare score più bassi a veri $\tau$/veri $b$-jets rispetto al caso (FT) (probabilmente comportamento ad hoc in caso di overlapping: le due reti sono addestrate per rigettare il fondo, e quando c'è overlapping con esso le caratteristiche associate distorcono il segnale).
        
- Coerentemente con quanto osservato sulla trasformazione della distribuzione degli score per il $\tau$ tagger GNTau: 
    - il caso (FT) che è maggioritario ($N=394332, \sim 99\%$ segnale $\tau$) segue la distribuzione uniforme
    - il caso (TT) **è fortemente minoritario** ($N=3617, <1\%$ del segnale) ha un picco per efficienze alte  
    ![](https://codimd.web.cern.ch/uploads/upload_69c541628b73cff0a580978e0b7823f7.png)
- Per il GN2 ($b$-tagger):
    ![](https://codimd.web.cern.ch/uploads/upload_22252e7c0312be93541379510591858a.png)

 **Commenti**
    - L'effetto dell'overlap tra due segnali reali ha una **ricaduta maggiore sulle performance** (livello di confidenza) **del tagger $\tau$ che del tagger $b$**


**RISULTATO - Le variabili legate al MET e al $\tau$ discriminano i casi true $\tau$ (_T) dai casi false $\tau$ (_F)**
Sfruttiamo le seguenti differenze tra il decadimento del $\tau$ adronico e gli eventuali processi in un (b)-jet per trovare informazione discriminante:
- un $\tau$ adronico decade come: $\tau \to$ adroni + $\nu_\tau$; poiché $m_\tau \approx 1.777 GeV$ e per il dataset considerato $p_T \sim 10^2 GeV$, i **prodotti di decadimento del $\tau$ risultano collimati lungo la sua direzione del moto** [DA FORMALIZZARE]
-> approssimativamente (*collinear approximation*, si veda 4.2 di [questo articolo](https://link.springer.com/article/10.1140/epjc/s10052-014-3120-z?utm_source=chatgpt.com)):
$$
\vec p_T ^{\nu_T} \simeq \alpha \vec p_T ^\tau, \qquad \alpha >0 
$$
nel sistema del laboratorio. 

- d'altra parte, in un jet potrebbero esserci neutrini (tramite decadimenti semileptonici degli adroni $B$) ma non necessariamente la loro direzione di emissione sarà correlata col quella del fake $\tau$ ricostruito

Considerando le seguenti variabili:
- ***MET*** lungo la direzione del candidato $\tau$:
$$
MET_{\parallel}
=
\vec E_T^{,miss}\cdot\hat p_T^\tau
=
E_T^{miss}\cos\left(\Delta\phi(\tau,E_T^{miss})\right),
$$
dove
$$
\Delta\phi(\tau,E_T^{miss})
=
\phi_\tau-\phi_{MET}
$$
riportato nell'intervallo $[-\pi,\pi]$.

    Questa variabile quantifica quindi direttamente quanto il vettore $E_T^{miss}$ sia allineato con la direzione del candidato $\tau$. Per un true $\tau$, il neutrino prodotto nel decadimento è approssimativamente collineare con gli adroni visibili, e ci si aspetta quindi
    $$
    \Delta\phi(\tau,E_T^{miss})\approx 0
    \qquad\Longrightarrow\qquad
    MET_{\parallel}>0.
    $$
    Per un fake $\tau$, invece, non è presente la stessa relazione cinematica tra il candidato ricostruito e la componente invisibile dell'evento; la distribuzione di $MET_{\parallel}$ risulta pertanto meno concentrata verso valori positivi.
    ![](https://codimd.web.cern.ch/uploads/upload_60af90b8979335d1cf71b556b116ac83.png)

- ***massa trasversa $m_T$***: 
$$
m_T
=
\sqrt{
2,p_T^\tau,E_T^{miss}
\left[
1-\cos\left(\Delta\phi(\tau,E_T^{miss})\right)
\right]
}.
$$

    La massa trasversa dipende esplicitamente dalla separazione azimutale tra il candidato $\tau$ e il MET. Nel limite collineare,
    $$
    \Delta\phi(\tau,E_T^{miss})\to 0,
    $$
    si ha
    $$
    1-\cos(\Delta\phi)\to0
    \qquad\Longrightarrow\qquad
    m_T\to0.
    $$
    Ci si aspetta quindi che gli eventi con un true $\tau$ siano maggiormente concentrati a piccoli valori di $m_T$. Per i fake $\tau$, l'angolo $\Delta\phi$ non è vincolato dalla stessa cinematica e può assumere valori più elevati, producendo una distribuzione di $m_T$ più estesa verso valori grandi.
![](https://codimd.web.cern.ch/uploads/upload_e393726dd7fa726504b7e59119086c85.png)

**RISULTATO - Potere discriminante delle variabili angolari assolute**
Le pseudorapidità $\eta$ sembrano essere distintive per la categoria FF:
![](https://codimd.web.cern.ch/uploads/upload_a45a4daade65a01ae8a0d9ecc82082d1.png)
![](https://codimd.web.cern.ch/uploads/upload_a0614e9ae54ecca1e5531ef2307d422c.png)

*NOTA*: non si è riscontrato potere discriminante per la variabile $\phi$ per $\tau$, jet

**RISULTATO - Potere discriminante delle variabili angolari relative**
Sembrano essere distintive per la categoria (FT):
[AGGIUNGI INTERPRETAZIONE]
![](https://codimd.web.cern.ch/uploads/upload_54aa24792ca6142c46b2808d7d4b7d90.png)
![](https://codimd.web.cern.ch/uploads/upload_6381d99eb9200b0c9a2644705de8ceac.png)

**RISULTATO - Potere discriminante del parametro geometrico relativo $\Delta R$**
Sembra essere distintiva per la categoria (FT):
[AGGIUNGI INTERPRETAZIONE]
![](https://codimd.web.cern.ch/uploads/upload_5dbf53bcb5e796dbcac90208fbd2319a.png)


---
## Lun 2026/09/21

### Attività
**Obiettivo 3.1**
- Studio delle caratteristiche degli oggetti reco overlappati e del loro potere discriminante, per le quattro categorie di verità:
    - rapporto tra il momento trasverso degli oggetti ricostruiti $p_{T, jet} / p_{T,\tau}$
    - numero di vertici primari ricostruiti `nPrimaryVertices`
    - score del tagger $\tau$ basato su RNN addestrato su background di jets: `tau_RNNJetScoreSigTrans` (si veda sez.5 [Reconstruction, Identification, and Calibration of hadronically decaying tau leptons with the ATLAS detector for the LHC Run 3 and reprocessed Run 2 data](https://cds.cern.ch/record/2827111/files/ATL-PHYS-PUB-2022-044.pdf))
    - score del tagger $\tau$ basato su RNN addestrato su background di elettroni:
`tau_RNNEleScoreSigTrans_v1` (si veda sez.6 [Reconstruction, Identification, and Calibration of hadronically decaying tau leptons with the ATLAS detector for the LHC Run 3 and reprocessed Run 2 data](https://cds.cern.ch/record/2827111/files/ATL-PHYS-PUB-2022-044.pdf))

**Obiettivo 3.2**
- Produzione delle metriche di valutazione associate ai $\tau$-tagger RNN `tau_RNNJetScoreSigTrans` e `tau_RNNEleScoreSigTrans_v1` separatamente sul dataset  $HH\to bb\tau\tau$:
    - ROC 
    - confusion matrix per soglia $\epsilon = 85\%$
    - efficiency vs score threshold
    - background rejection vs efficiency
    - score distribution

- Studio della correlazione tra score del $b$-tagger GN2 e score dati da $\tau$-tagger (`tau_GNTauScoreSigTrans_v0prune`, `tau_RNNJetScoreSigTrans` e `tau_RNNEleScoreSigTrans_v1`), al variare delle quattro categorie di verità:
    - Produzione violin plot (categorie discrete quantili GN2 VS score continui $\tau$-taggers)
    - Calcolo del *[correlation ratio](https://en.wikipedia.org/wiki/Correlation_ratio)* $\eta$ -> indice di dispersione degli score continui sul $\tau$ per alcune categorie di score di GN2 (quantili discreti) rispetto alla media:
    $$
    \eta^2 := \frac{Var_X(score_\tau |X)}{Var(score_\tau)}
    $$
    dove gli indici $X$ rappresentano le diverse categorie discrete (quantili) associate agli score di GN2. 
    Si ha $0\le \eta \le 1$, dove:
        - $\eta \simeq 0$: le medie di $score_\tau$ sono sostanzialmente invariate tra i quantili -> poca dipendenza da GN2
        - $\eta \simeq 1$: i valori di GN2 separano quasi completamente gli score dei tagger su $\tau$ -> forte dipendenza
    - Calcolo del *[coefficiente di Spearmann](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)* $\rho_s$, escludendo le categorie di tagging invalidi (`-1` per GN2) -> indice di monotonia della relazione tra gli score

  
### Risultati

**RISULTATI - Potere discriminante del rapporto $p_{T, jet} / p_{T,\tau}$**
La feature ricavata è discriminante per il caso (FT) (*Trust Tau*) rispetto alle altre categorie di verità; una possibile interpretazione per cui $p_{T,jet} > p_{T,\tau}$ sistematicamente nei casi (T_) (*Trust jet*) è che in questi casi il falso $\tau$ è probabilmente ricostruito a partire da un sottoinsieme delle tracce presenti nel jet reale.
![](https://codimd.web.cern.ch/uploads/upload_85f5fc32d573948e8bf7857eddd4e0ea.png)


**OSSERVAZIONE - Il $\tau$-tagger per elecron discrimination `tau_RNNEleScoreSigTrans_v1` ha distribuzioni di score diverse rispetto ai tagger `tau_RNNJetScoreSigTrans`, `tau_GNTauScoreSigTrans_v0prune`**
Runnando lo stesso codice già usato per `tau_GNTauScoreSigTrans_v0prune` sulla distribuzione dello score per le quattro categorie di verità (previa identificazione dei working points degli score associate alle efficienze $\epsilon_\tau$ di interesse), si trova che:
- `tau_RNNJetScoreSigTrans` ha distribuzioni sostanzialmente equivalenti a quelle di `tau_GNTauScoreSigTrans_v0prune`:
    - trasformazione su segnale -> distribuzione uniforme del caso (FT) lungo lo score
    - stesso deterioramento delle prestazioni (rejection di vero segnale) per il caso minoritario (TT)
    - comportamento atteso su falsi $\tau$ (casi (FF), (TF))
![](https://codimd.web.cern.ch/uploads/upload_29c3d500da4e60d5811b46f6fc5eab38.png)

- `tau_RNNEleScoreSigTrans_v1` ha invece un comportamento diverso, derivante dal diverso problema fisico che è allenato a discriminare (elettroni VS $\tau$ adronici, al posto di $b$-jets VS $\tau$ adronici):
    - distribuzione (pseudo?) uniforme di FT (*trust Tau*) sull'asse dello score
    - il tagger fallisce ad identificare correttamente il fake $\tau$ nella maggior parte dei casi per (TF, FF), con una caratteristica distribuzione a due picchi dello score (rigetta con certezza oppure accetta con certezza)
    - stessa distribuzione anche per il caso (TT)
-> per questo problema fisico, questo tagger **NON** aggiunge informazione complementare utile per discriminare i casi non discriminati dagli altri due tagger (i casi (TF, TT, FF)): il comportamento su queste tre categorie è sostanzialmente lo stesso

    ![](https://codimd.web.cern.ch/uploads/upload_4315572d125d504018ea512d0ade1ad2.png)

**OSSERVAZIONE - Il potere discriminante dell'informazione congiunta [tau score $\times$ b-jet score] è alto per le categoria (TT) VS (FT)**
```
====================================================================================================
CORRELAZIONE (score tau) vs (quantile bin GN2) - per categoria di verità
====================================================================================================
                          (a) TT              (b) TF              (c) FT              (d) FF
tagger                    eta     rho_s        eta     rho_s        eta     rho_s        eta     rho_s
GNTauScoreSigTrans_v0prune  0.413  -0.419       0.172  -0.112       0.066  +0.011       0.055  +0.012
RNNJetScoreSigTrans         0.386  -0.343       0.143  -0.020       0.087  -0.064       0.051  +0.020
RNNEleScoreSigTrans_v1      0.134  -0.148       0.021  -0.053       0.073  -0.077       0.038  -0.025
```
- GNTau:
    ![](https://codimd.web.cern.ch/uploads/upload_1a2d50768c7d607b7ecb7c69a6e3c4fa.png)

- RNNJet:
![](https://codimd.web.cern.ch/uploads/upload_ee4a3d61b3cd20f9980635ef73bafe07.png)

- RNNEle:
![](https://codimd.web.cern.ch/uploads/upload_4e8b84335883a64f2a7d885001708d92.png)


**Commenti**
- Per tutti e tre i $\tau$-tagger, l'assenza di correlazione nel caso (FT) (*Trust Tau*) è attesa: gli score dei $\tau$ sono distribuiti uniformemente su ogni bin di score di GN2, coerentemente col fatto che gli score dei $\tau$ sono *signal transformed*
- Per i due tagger più fisicamente rilevanti al problema (`GNTauScoreSigTrans_v0prune` e `RNNJetScoreSigTrans `), la correlazione (negativa) tra quanto un jet è marcatamente b-taggato (bin di quantile crescente) e lo score del $\tau$ candidato è sistematicamente più forte nella categoria (TT) rispetto alle altre tre categorie: $\eta \sim 0.4, \rho_2 \sim -0.4$ per (TT), contro valori quasi nulli
->  **mantenere almeno un tagger per $\tau$ tra `GNTauScoreSigTrans_v0prune` e `RNNJetScoreSigTrans` insieme a GN2 come feature nella DNN è una scelta motivata, perché l'andamento reciproco di questi score discrimina il caso (TT) dal caso (FT)**
- `RNNEleScoreSigTrans_v1` mostra una correlazione sistematicamente più debole rispetto agli altri due tagger anche in TT, coerente con quanto già osservato sulla sua scarsa capacità di distinguere le categorie TF/TT/FF: essendo ottimizzato per un problema fisico diverso (rigetto elettroni), non condivide lo stesso meccanismo di degrado reciproco jet-τ, e **resta la feature meno prioritaria dei tre τ-score da considerare per la DNN**.


---
## Mar 2026/09/22

### Attività
**Obiettivo 3.3**
- Preparazione dei dataset per train/validation/test della DNN:
    - addattamento ed espansione del codice della precedente repo `flavour_tagging` al preprocessing di file `.root`
    - estrazione delle feature di interesse e creazione dei file `.npz` associati
    - splitting train/val/test **con coppie overlap dello stesso evento rigorosamente nello stesso dataset**

### Risultati
--

---
## Mer 2026/09/23

### Attività
**Obiettivo 3.3**
- Implementazione delle metriche `F1_score` e `ap_mean` come alternative alla media/mediana della validation loss per la valutazione del best model e dell'early stopping, dove dette
$$
\text{Precision} := \frac{TP}{TP + FN}, \qquad \text{Recall} := \frac{TP}{TP + FP}
$$
si ha
    - `F1_score`: media armonica tra precision e recall
    $$
    F1 := 2\cdot \frac{P\cdot R}{P+R}
    $$
    - `ap_mean` è la media (su tutte le classi) dell'*average precision*, ovvero l'area sotto la curva $\text{Precision}(\text{Recall})$. 

- Training e evaluation DNN su datasets $HH\to bb\tau\tau$:
    - run con weighted loss (`CrossEntropyLoss`) + no sampling strategy 
    - run con weighted loss (`CategoricalFocalLoss`) + no sampling strategy
dove i pesi assegnati alle categorie sono proporzionali all'inverso delle frequenze delle categorie (normalizzate sul numero di classi):
```python
  train_class_counts = torch.bincount(torch.as_tensor(y_train, dtype=torch.long), minlength=NUM_CLASSES).float()
  CLASS_WEIGHTS = train_class_counts.sum() / (NUM_CLASSES * train_class_counts)
```
ovvero:
$$
\omega_c = \frac N{K N_c}
$$
dove $K$ è il numero di classi.

In questi training, il best model sul validation loss è valutato usando 

### Risultati

**APPROFONDIMENTO - `Cross-Entropy Loss` VS Weighted loss (`Balanced Cross-Entropy Loss` e `Categorical Focal Loss`)
Come spiegato in questo blog [Focal Loss : A better alternative for Cross-Entropy](https://towardsdatascience.com/focal-loss-a-better-alternative-for-cross-entropy-1d073d92d075/), la Cross Entropy (`nn.CrossEntropyLoss()`) 
$$
CE = -\sum_{i=1}^n Y_i \log(p_i)
$$
(dove $Y_i$ è la truth label associata a $i$, quindi $p_i$ è lo score associato alla corretta classificazione di $i$)
performa male in caso di class imbalance in quanto:
1. non affronta il problema classi numerose VS classi rare: le classi con molteplicità maggiore spingeranno la loss verso minimi ottimizzati **per loro**
2. non affronta il problema classi facili VS classi difficili (dove per *difficili* si intende classi dove il modello sbaglia spesso e tanto nella classificazione)

Per risolvere questi due problemi, rispettivamente:
1. -> introduzione pesi $\alpha_i$, inversamente proporzionali alle numerosità delle classi secondo qualche criterio -> *Balanced Cross-Entropy Loss*:
$$
Balanced CE = -\sum_{i=1}^n \alpha_i \log(p_i)
$$
2. -> introduzione fattori modulanti $(1-p_i)^\gamma$, che riducono il peso degli esempi ben classificati ($p_i\to 1$, con $\gamma \ge 1$), al crescere di $\gamma$ -> *Categorical Focal Loss*:
$$
CF = -\sum_{i=1}^n (1-p_i)^\gamma \log(p_i)
$$

--> mettendo insieme il meglio dei due mondi si ottiene la versione pesata della CF:
$$
BCF = - \sum_{i=1}^n \alpha_i (1-p_i)^\gamma \log(p_i)
$$


 
**RISULTATO - La `CategoricalFocalLoss` modifica significativamente il comportamento sulla classe TT, ma non migliora globalmente le prestazioni**

Compariamo i due training effettuati con gli stessi iperparametri seguenti

ma rispettivamente usando:
- [`criterion = CrossEntropyLoss()`](https://github.com/giumont/overlap_resolver/tree/main/outputs/DNN/HH_bbtt_dataset/2layers_32_16_dropout0.1_LayerNorm_batch2048_lr0.001_wd0.0001_best_on_ap_macro_criterion_CrossEntropyLoss()_weight_loss_True_sampling_strategy_none)
    ![](https://codimd.web.cern.ch/uploads/upload_dd5d4223a8b79bd59623c040237f0b38.png)
    ![](https://codimd.web.cern.ch/uploads/upload_df42d237263180415d68bc65a798ef56.png)
    ![](https://codimd.web.cern.ch/uploads/upload_effa585ea0c95397794c108f10b64fbe.png)
con:
```
Risultati - Train set (multiclasse)
-----------------------------------
  n_totale: 376832
  accuracy: 94.91%
  classe  precision     recall         f1    support
      FF     81.63%     82.14%     0.8188      38075
      FT     97.31%     98.60%     0.9795     275529
      TF     92.20%     90.02%     0.9110      60764
      TT      0.00%      0.00%     0.0000       2464
  macro avg   - precision: 67.78%  recall: 67.69%  f1: 0.6773
  weighted avg- precision: 94.26%  recall: 94.91%  f1: 0.9458

Risultati - Test set (multiclasse)
----------------------------------
  n_totale: 81083
  accuracy: 94.81%
  classe  precision     recall         f1    support
      FF     81.28%     82.76%     0.8201       8140
      FT     97.34%     98.52%     0.9793      59204
      TF     91.75%     89.96%     0.9084      13125
      TT      0.00%      0.00%     0.0000        614
  macro avg   - precision: 67.59%  recall: 67.81%  f1: 0.6770
  weighted avg- precision: 94.08%  recall: 94.81%  f1: 0.9444
  ```
    
- [`criterion = CategoricalFocalLoss(alpha=CLASS_WEIGHTS), gamma=2.0`](https://github.com/giumont/overlap_resolver/tree/main/outputs/DNN/HH_bbtt_dataset/2layers_32_16_dropout0.1_LayerNorm_batch2048_lr0.001_wd0.0001_best_on_ap_macro_criterion_CategoricalFocalLoss()_weight_loss_True_sampling_strategy_none) (valori di default)
    ![](https://codimd.web.cern.ch/uploads/upload_0227967aeaf3b0eb37f4e7d5fab02189.png)
    ![](https://codimd.web.cern.ch/uploads/upload_1fa6b09f46c00290bf23c51657387673.png)
    ![](https://codimd.web.cern.ch/uploads/upload_6128a835e8bf751749e65a68a89890de.png)
con:
```
Risultati - Train set (multiclasse)
-----------------------------------
  n_totale: 376832
  accuracy: 87.80%
  classe  precision     recall         f1    support
      FF     67.01%     90.97%     0.7717      38063
      FT     99.02%     92.82%     0.9582     275538
      TF     95.98%     64.73%     0.7732      60767
      TT      4.37%     45.94%     0.0798       2464
  macro avg   - precision: 66.60%  recall: 73.62%  f1: 0.6457
  weighted avg- precision: 94.68%  recall: 87.80%  f1: 0.9038

Risultati - Test set (multiclasse)
----------------------------------
  n_totale: 81083
  accuracy: 87.68%
  classe  precision     recall         f1    support
      FF     66.60%     91.31%     0.7703       8140
      FT     99.01%     92.80%     0.9580      59204
      TF     95.61%     64.43%     0.7698      13125
      TT      4.69%     42.67%     0.0845        614
  macro avg   - precision: 66.48%  recall: 72.80%  f1: 0.6457
  weighted avg- precision: 94.49%  recall: 87.68%  f1: 0.9021
  ```
  
 **Commenti**

- Usare la loss pesata **sembra comunque non sufficiente mantenendo la `CrossEntropyLoss`**: nonostante il peso assegnato alla classe TT sia molto elevato (`w_TT = 38.2`), il modello continua a non assegnare nessun evento alla classe TT secondo la decisione `argmax`. Sul test set si ottiene infatti `recall(TT) = 0` e `precision(TT) = 0`.

- Passando alla **`CategoricalFocalLoss` con `gamma=2`**, la situazione della classe TT cambia qualitativamente: il modello inizia ad assegnare eventi a TT e raggiunge sul test set `recall(TT) = 42.67%`. Lo stesso comportamento e' presente anche sul train set (`recall = 45.94%`), quindi il recupero della classe TT non appare limitato a un comportamento specifico del training set.

- Questo risultato rappresenta un cambiamento qualitativo importante: la Focal Loss permette al modello di produrre predizioni `TT` non nulle, mentre la sola pesatura della Cross Entropy non era sufficiente a far si' che TT vincesse il confronto tra gli score in `argmax`.

- Tuttavia, **il recupero della recall TT avviene al prezzo di una perdita significativa sulle altre classi**. Sul test set:

| Classe | CE recall | Focal recall | CE precision | Focal precision | CE F1 | Focal F1 |
|---|---:|---:|---:|---:|---:|---:|
| FF | 82.76% | 91.31% | 81.28% | 66.60% | 0.8201 | 0.7703 |
| FT | 98.52% | 92.80% | 97.34% | 99.01% | 0.9793 | 0.9580 |
| TF | 89.96% | 64.43% | 91.75% | 95.61% | 0.9084 | 0.7698 |
| TT | 0.00% | 42.67% | 0.00% | 4.69% | 0.0000 | 0.0845 |

In particolare, la classe **TF perde circa 25 punti percentuali di recall**, mentre FF aumenta la propria recall ma perde una parte consistente di precision. La Focal Loss non produce quindi un miglioramento uniforme: **modifica il compromesso tra le diverse classi**.

- Questo e' visibile anche nelle metriche aggregate. La `accuracy` passa da `94.81%` a `87.68%` e il `macro F1` da `0.6770` a `0.6457`. Quindi, **secondo queste metriche non c'e' un miglioramento globale del classificatore**. Il vantaggio della Focal Loss e' invece specificamente legato alla capacita' di ottenere una predizione non nulla per la classe TT.

- E' importante inoltre considerare la **precision estremamente bassa ottenuta per TT**. Sul test set:

```text
TP ~= 0.4267 * 614 ~= 262
```
mentre dalla precision:
```text
N_predicted_TT `= 262 / 0.0469 ~= 5.6 * 10^3
```
Quindi circa 262 degli eventi classificati come TT sono realmente TT, mentre la grande maggioranza delle predizioni TT e' costituita da falsi positivi. Il recupero della recall, quindi, **non corrisponde ancora a una buona purezza della selezione TT**.


---
## Gio 2026/09/24

### Attività
**Obiettivo 3.3**
- Scan sui parametri caratteristici $(p_\alpha, \gamma)$ associati alla loss `CategoricalFocalLoss()`, dove (con $\alpha$ inverso delle frequenze relative delle classi):
    - $p_\alpha$ è la potenza dei pesi relativi delle classi:  `CLASS_WEIGHTS = alpha**alpha_p` ($0<p_\alpha \le 1$ per ottenere classi piÙ rare che hanno piÙ peso nella loss)
    - $\gamma$ definisce il peso dei fattori modulanti (classi facili vs difficili) $(1-p_i)^\gamma$ per questa loss


### Risultati

**RISULTATO - Scan performance sui parametri caratteristici della focal loss**

- metrica: average precision generale del modello (media su tutte le classi):
![](https://codimd.web.cern.ch/uploads/upload_24e811d402af5e891ce961549d151f75.png)

- metrica: average precision del modello per la classe TT (classe critica):
![](https://codimd.web.cern.ch/uploads/upload_b612008b09acc91e52a5be236c249c5f.png)

**Commenti**
- **Scelta migliore dei parametri**: la combinazione ottimale si trova nella zona a basso $\gamma$, in particolare $\gamma=0$–$1$ con $p_\alpha \approx 0.25$–$0.5$. In questa regione si ottiene sia il massimo di `val_ap_macro` (**0.722**, per $\gamma=0$ e $\gamma=1$ con $p_\alpha=0.25$–$0.5$) sia il massimo di `val_ap_TT` (**0.062**, per $\gamma=1$, $p_\alpha=0.5$), suggerendo che il punto $(\gamma=1,\ p_\alpha=0.5)$ rappresenti il miglior compromesso complessivo tra le due metriche.

- **Interpretazione**: l'andamento sulla griglia è chiaro e *monotono* in $\gamma$: all'aumentare del fattore di modulazione $(1-p_i)^\gamma$ le prestazioni peggiorano in modo sistematico, sia per `val_ap_macro` che, in modo ancora più marcato, per `val_ap_TT`. Poiché $\gamma=0$ annulla il termine modulante e riduce la loss a una semplice *cross-entropy pesata* da $\alpha^{p_\alpha}$, il risultato indica che il beneficio della focal loss (dare più peso agli esempi difficili) è qui controproducente: per la classe TT, già enfatizzata dal reweighting $\alpha$, l'aggiunta della modulazione $\gamma$ produce un "doppio focusing" che concentra il gradiente su un sottoinsieme troppo ristretto di esempi difficili, rendendo l'ottimizzazione più rumorosa e instabile invece di migliorare la separazione tra classi. Anche il parametro $p_\alpha$ mostra un comportamento non monotono: valori intermedi (0.25–0.5) sono sistematicamente migliori sia di $p_\alpha=0$ (nessun reweighting, sottopesa le classi rare) sia di $p_\alpha=1$ (reweighting completo, che penalizza eccessivamente le classi maggioritarie a scapito della metrica media).

- **Rilevanza della differenza tra le metriche**: è importante notare la differenza di scala tra le due griglie. Su `val_ap_macro` la variazione totale è molto contenuta (0.710–0.722, circa l'**1.7%** relativo), un intervallo compatibile con il rumore statistico intrinseco del training (inizializzazione, batching, ecc.), per cui le differenze tra le celle vicine non sono necessariamente significative. Su `val_ap_TT`, invece, la variazione è molto più ampia (0.035–0.062, circa il **77%** relativo tra minimo e massimo), un effetto ben oltre il rumore atteso: questo conferma che la scelta di $(\gamma, p_\alpha)$ ha un impatto reale e rilevante specificamente sulla classe critica TT, mentre il suo effetto sulla performance media del modello è marginale. Questo rafforza la conclusione che **la focal loss, più che migliorare le prestazioni generali, rischia soprattutto di destabilizzare l'apprendimento sulla classe di interesse quando $\gamma$ viene spinto verso valori elevati**

Per la [scelta ($p_\alpha = 0.5, \gamma = 1$)](https://github.com/giumont/overlap_resolver/tree/main/outputs/DNN/HH_bbtt_dataset/SWEEP_FOCAL_PARAMS_2layers_32_16_dropout0.1_LayerNorm_batch2048_lr0.001_wd0.0001_best_on_ap_macro_criterion_CategoricalFocalLoss()_weight_loss_True_sampling_strategy_none/focal_a0.5_g1) si ottengono i seguenti risultati: 
    ![](https://codimd.web.cern.ch/uploads/upload_c383b29df4aa0a4dbbce6e02eb3e4e27.png)
    ![](https://codimd.web.cern.ch/uploads/upload_2642383bc77f307b6a84a7176418efcc.png)
    ![](https://codimd.web.cern.ch/uploads/upload_c8810d4d97337a24a0e171366005c200.png)


---
## Ven 2026/09/25

### Attività
**Obiettivo 3.1**
- Indagine: conteggi delle classi FF e TT 
    - al diminuire della soglia geometrica di overlap $\Delta R$
    - implementando algoritmo greedy per coppie (reco tau, reco jet) **senza ripetizione** per ogni evento:
        1. calcola il valore di $\Delta R$ per tutte le coppie possibili presenti nell'evento
        2. sceglie le coppia (jet reco, tau teco) con il minore $\Delta R$
        3. Se $\Delta R$ < soglia di overlapping (fissata a priori) per la coppia in questione, essa viene salvata e messa da parte
        4. Ripete il punto 2 e 3  con le coppie rimanenti dopo questa rimozione

**Obiettivo 3.3**
- Studio architettura Transformer:
    - [Overview of functionality](https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452/)
    - [How it works, step by step](https://towardsdatascience.com/transformers-explained-visually-part-2-how-it-works-step-by-step-b49fa4a64f34/)
    - [Multi-head attention](https://medium.com/data-science/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853)
- Implementazione architettura trasformer in alternativa a DNN per riconoscimento classi



### Risultati

**OSSERVAZIONE - Irrobustire i criteri sulla scelta delle coppie in overlapping non elimina i conteggi legati alle classi FF e TT nel dataset di segnale $HH\to bb\tau\tau$**
Implementando l'algoritmo per la scelta senza ripetizione degli elementi nelle coppie in overlapping, il numero di coppie della categoria "True + True" non scende significativamente rispetto al metodo utilizzato fino ad ora per fare le coppie (ovvero prendendo tutte le coppie con $\Delta R$ < soglia fissata, senza preoccuparsi di eventuali ripetizioni).

Per la categoria "True + True" al variare della soglia di $\Delta R$, per i due diversi metodi di scelta delle coppie:

```
- deltaR_max = 0.4: 3617 (con possibili ripetizioni) vs 3587 (senza ripetizioni)
- deltaR_max = 0.2: 3547 vs 3546
- deltaR_max = 0.045: 2681 vs 2681
```

Risultati simili si ottengono per la categoria "False + False". Sembra effettivamente esserci una quota irriducibile di coppie in overlapping per queste due categorie. 



--- 
## Sab 2026/09/26


### Risultati
**OSSERVAZIONE - Le categorie minoritarie per il fondo $t\bar t$ sono quelle contenenti $\tau$ adronici, e le proporzioni osservate sono compatibili come ordine di grandezza (stima non precisa) con i BR attesi**

In questo caso, **le categorie minoritarie non sono (TT) e (FF), bensì (TT) e (FT)**. Questo ha senso, perchè in questo dataset completamente adronico (da $t\bar t \to Wb$, si considerando solo decadimenti adronici $W\to q\bar q'$) gli unici veri $\tau$ possono provenire dai decadimenti in volo degli adroni pesanti come 
$$
B\to \tau \nu_\tau X, 
$$
originatosi da beauty, oppure
$$
D_s \to \tau \nu_\tau 
$$
originatosi da charm + antistrange.
**Anche in questo caso una quota relativa a queste categorie minoritarie sembra irriducibile**, confrontando i risultati ottenuti per i due diversi algoritmi di creazione delle coppie in overlapping (si vedano conteggi nelle due immagini).

- Con possibili ripetizioni
![](https://codimd.web.cern.ch/uploads/upload_a04b564d0aaf662a5c838eb945c35d74.png)

- Senza ripetizioni
![](https://codimd.web.cern.ch/uploads/upload_8b0c105d470bd47a486579e10a65915c.png)

Come visibile dai conteggi delle immagini, le proporzioni relative delle quattro diverse classi sono (considerando qui i casi con ripetizioni, differenze trascurabili):
$N_{tot} = 102\,468$ coppie con $\Delta R < 0.4$, con:

| Classe | Descrizione      |      N | Frazione |
| ------ | ---------------- | ------:| --------:|
| TF     | b vero, tau fake | 24 896 |  24.30 % |
| FT     | b fake, tau vero |     72 |  0.070 % |
| TT     | entrambi veri    |    632 |  0.617 % |
| FF     | entrambi fake    | 76 868 |  75.02 % |

Da queste si ricavano due numeri utili:

- **Coppie con jet b:** 24.9 %
- **Probabilità che il tau sia vero, data la coppia:**
  - se il jet è b: 632 / 25 528 ≈ **2.5 %**
  - se il jet non è b: 72 / 76 940 ≈ **0.094 %**
  - il rapporto tra le due è circa **26**

Possiamo confrontare questi valori con le previsioni da branching ratio:

- **Se il jet è b (TT):** il $\tau_{\text{had}}$ vero atteso è dell'ordine dell'**1.6 %** per jet.
INFATTI: 

    Il processo $b \to \tau$ avviene tramite due vie:

    - Decadimento inclusivo (diretto + indiretto tramite c): $\text{BR}(b \to \tau^- \bar{\nu}_\tau X) \approx 2.4\%$ (ad esempio, in [ALEPH](https://arxiv.org/pdf/hep-ex/0010022))
    - Frazione adronica del $\tau$: $\approx 65\%$ ($\approx 64.8\%$ dal [PDG](https://pdg.lbl.gov/2025/reviews/rpp2025-rev-tau-branching-fractions.pdf?utm_source=chatgpt.com))

    $$
    \text{P}(\tau_{\text{had}} \text{ vero in jet } b) = 2.4\% \times 65\% \approx 1.56\% \approx 1.6\%
    $$

- **Se il jet non è b (FT):** il $\tau_{\text{had}}$ vero atteso è dell'ordine dello **0.05 %** per jet.
INFATTI
La sorgente principale di $\tau$ genuini in jet non-$b$ è il decadimento del mesone $D_s$ prodotto dalla frammentazione del quark $c$ originato da $W \to c\bar{s}$:

    - Frazione di jet di $c$ tra i jet non-$b$: $\approx 20\%$ [CALCOLARE DAI DATI]
    - Probabilità di frammentazione $c \to D_s$: $\approx 8\%$ ([fonte](https://link.springer.com/article/10.1140/epjc/s10052-016-4246-y?utm_source=chatgpt.com))
    - Decadimento del $D_s$: $\text{BR}(D_s \to \tau \nu) \approx 5.33\%$ (dal [PDG](https://pdg.lbl.gov/2025/reviews/rpp2025-rev-pseudoscalar-meson-decay-cons.pdf?utm_source=chatgpt.com))
    - Frazione adronica del $\tau$: $\approx 65\%$ ($\approx 64.8\%$ dal [PDG](https://pdg.lbl.gov/2025/reviews/rpp2025-rev-tau-branching-fractions.pdf?utm_source=chatgpt.com))

    $$
    \text{P}(\tau_{\text{had}} \text{ vero in jet non-}b) = [20\%] \times 8\% \times 5.4\% \times 65\% \approx 0.056\% \approx 0.05\%
    $$

<!--
## Confronto con i dati

### Rapporto tra jet $b$ e non-$b$

Il rapporto tra i tassi a priori è

$$
\frac{1.9\%}{0.056\%} \approx 34
$$

mentre nei dati il rapporto tra le probabilità $\text{P}(\tau \text{ vero} \mid \text{coppia})$ è circa **26**. I due valori sono dello stesso ordine di grandezza. La differenza è attesa perché nei dati compaiono anche l'efficienza di ricostruzione del $\tau$ vero e il tasso di candidati tau fake, che possono differire tra jet $b$ e jet non-$b$. Dopo la ricostruzione, ad esempio, i $\tau$ da decadimento di adroni $b$ sono in media meno isolati.

### Tasso di fake implicito

La probabilità osservata di avere un tau vero, data la coppia, è

$$
p_{\text{oss}} \approx \frac{\varepsilon_{\text{reco}} \cdot \text{P}_{\text{vero}}}{\varepsilon_{\text{reco}} \cdot \text{P}_{\text{vero}} + f_{\text{fake}}} \approx \frac{\varepsilon_{\text{reco}} \cdot \text{P}_{\text{vero}}}{f_{\text{fake}}}
$$

dove $\varepsilon_{\text{reco}}$ è l'efficienza di ricostruzione del $\tau_{\text{had}}$ vero e $f_{\text{fake}}$ è il numero di candidati tau fake per jet. Invertendo la relazione con $\varepsilon_{\text{reco}} \approx 0.3\text{–}0.5$:

| Jet | $p_{\text{oss}}$ | $\text{P}_{\text{vero}}$ | $f_{\text{fake}}$ implicito |
| --- | ----------------:| ------------------------:| ---------------------------:|
| $b$ | 2.5 % | 1.9 % | $\approx 23\text{–}38\,\%$ |
| non-$b$ | 0.094 % | 0.056 % | $\approx 18\text{–}30\,\%$ |

Le due stime indipendenti danno un tasso di fake dello stesso ordine, circa **20–35 % di candidati tau fake per jet**. È plausibile se i tau "analysis" sono solo preselezionati, senza ID, perché molti jet hanno 1 o 3 tracce nel core.

### Limiti della stima

- $\varepsilon_{\text{reco}}$ è assunta, non misurata: i risultati dipendono da questa scelta.
- I tassi a priori sono ordini di grandezza, non calcoli precisi. La frazione di jet di $c$ (20 %) e la probabilità $c \to D_s$ (8 %) sono incerte.
- La classe FT ha solo 72 coppie, quindi $p_{\text{oss}}$ per i jet non-$b$ ha un errore statistico di circa il 12 %.
- Parte dei tau veri in FT può stare in jet non etichettati $b$ per come è definito il truth label, ad esempio un adrone $b$ sotto soglia di $p_T$ o fuori dal cono.
- Il controllo più diretto è guardare l'antenato (PDG ID) dei $\tau$ veri in TT e FT: devono venire da adroni $B$ o $D_s$, non da $W$.
-->

**OSSERVAZIONE - Le feature discriminanti sul dataset $HH\to bb\tau\tau$ non sembrano esserlo altrettanto sul fondo $t\bar t$**

Ripetendo le analisi sulle stesse feature considerate per il dataset del segnale, stavolta sul fondo, non si ottengono le stesse distribuzioni discriminanti tra le categorie di veritÀ delle coppie in overlapping. Questo è coerente con l'osservazione che i processi fisici che l'algoritmo dovrebbe classificare nella stessa categoria (es: (FF)) non sono necessariamente gli stessi tra i due dataset, che simulano eventi differenti. 

- Esempio 1: rapporto tra i momenti trasversi del jet reco e del tau reco:
    - Segnale $HH\to bb\tau\tau$
    ![](https://codimd.web.cern.ch/uploads/upload_db64087b6410428aa2deb51e9da4b480.png)
    - Fondo $t\bar t$
    ![](https://codimd.web.cern.ch/uploads/upload_8bf2d8b2af2112448702b679265c4ab5.png)

- Esempio 2: parametri angolari del jet/del tau reco:
    - Segnale $HH\to bb\tau\tau$
    ![](https://codimd.web.cern.ch/uploads/upload_ba626402a5a10e8ba822f0ffd02952e7.png)
    - Fondo $t\bar t$
    ![](https://codimd.web.cern.ch/uploads/upload_6e2160258f9c19924692775adea56ac7.png)


In molti altri casi, la poca statistica per le categorie (_T) sul dataset di fondo rende difficile comparare le distribuzioni col caso del segnale (sebbene una ricerca/maggiore conoscenza della fisica del problema potrebbe aiutare a prevedere il comportamento atteso):
- Esempio 1: momenti trasversi
    - Segnale $HH\to bb\tau\tau$
    ![](https://codimd.web.cern.ch/uploads/upload_64fb64790eb7bd799b6542c481fc851f.png)
    - Fondo $t\bar t$
    ![](https://codimd.web.cern.ch/uploads/upload_e985f33040abbf559c693889b9eebb75.png)

- Esempio 2: variabili angolari relative
    - Segnale $HH\to bb\tau\tau$
    ![](https://codimd.web.cern.ch/uploads/upload_58d05af1e20b8bc87c0275e7e798bd72.png)
    - Fondo $t\bar t$
    ![](https://codimd.web.cern.ch/uploads/upload_8e7a2ac585c466adb6baa61882010a9b.png)

- Esempio 3: $\Delta R$:
    - Segnale $HH\to bb\tau\tau$
    ![](https://codimd.web.cern.ch/uploads/upload_8ee9c3da64c119ff772e2c1e86933dd3.png)
    - Fondo $t\bar t$
    ![](https://codimd.web.cern.ch/uploads/upload_45f5582ccb9e63624e792834a788f3a6.png)

- Esempio 4: decadimento del $\tau$ reco:
    - Segnale $HH\to bb\tau\tau$
    ![](https://codimd.web.cern.ch/uploads/upload_4d00fec6ddcaf3d08ed8cd5add8bb5d7.png)
    - Fondo $t\bar t$
    ![](https://codimd.web.cern.ch/uploads/upload_34910029e2fb9847a96c0fa0dfd04252.png)
e:
    - Segnale $HH\to bb\tau\tau$
    ![](https://codimd.web.cern.ch/uploads/upload_5872d0c33571b850ceb762a9fb5ccb79.png)
    - Fondo $t\bar t$
    ![](https://codimd.web.cern.ch/uploads/upload_d243f258290e075563580d02767c4a9d.png)
    
    
    

**OSSERVAZIONE - Nel fondo adronico $t\bar t$, gli eventi (_T) non sono associati a score alti dei tagger $\tau$**
Per il dataset di segnale $HH$, si era trovato che:
- la categoria (FT), fortemente maggioritaria tra le categorie (_T) (vero $\tau$ adronico) presentava score dei tagger $\tau$ GNTau e della controparte RNN distribuiti uniformemente (*signal transformed*)
- la categoria (TT), tuttavia, presentava score associati molto bassi (spesso $\tau$ scartati per working points con $\epsilon_\tau) < 97\%$); questo comportamento potrebbe essere dovuto alla sovrapposizione con veri b-jet, che sembra peggiorare le performance dei tagger per entrambi gli oggetti

Per questo dataset di fondo $t\bar t$, quello che si nota invece è che **tutti gli eventi (_T)** (`isHadronicTau==True`) **NON vengono riconosciuti dai tagger dei $\tau$**. 
Per il tagger a RNN (e similmente per GNTau):
![](https://codimd.web.cern.ch/uploads/upload_0520c93928290f2d7ac3f2dd293f87f7.png)

Al contrario, **la discrepanza delle performance del tagger dei jet GN2 nel caso (TT) rispetto al caso (TF) sembra inferiore rispetto al dataset di segnale**:
![](https://codimd.web.cern.ch/uploads/upload_d75836d780ee4ad3837d3a4f0b3eeb8c.png)

Anche in questo caso, il tagger $\tau$ RNNEle, pensato per un problema di identificazione diverso, fallisce nel discriminare le classi: 
![](https://codimd.web.cern.ch/uploads/upload_e2336852de30847896cfa70ab0cdbcb2.png)

**Commenti**
- Questo comportamento è compatibile col fatto che gli unici $\tau$ presenti in questo dataset sono **parte di catene di decadimento lunghe degli adroni e non sono isolati**, infatti i tagger $\tau$ sono addestrati per distinguere dai jet adronici i $\tau$ *prompt* (questo andrebbe approfondito cercando nella documentazione i dataset su cui sono addestrati i tagger)

---
## Dom 27/09/2026, Lun 28/09/2026

**Attività**
- Aggiornamento logbook con progressi

--- 
## Mar 29/09/2026

**Attività**
- Burocrazia