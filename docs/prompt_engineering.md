# Prompt Engineering -- Journal de bord

Ce document retrace comment on a utilise des agents IA (Claude, ChatGPT,
Copilot) pour developper PediAppend. On est 5 dans l'equipe. Le principe :
on prend la tache, on cree les dossiers et les fichiers vides, et l'agent
IA implemente directement dans le code. On relit tout avant de merge.

---

## Table des matieres

1. [Workflow general](#1-workflow-general)
2. [Phase 1 -- Exploration des donnees (EDA)](#2-phase-1----exploration-des-donnees-eda)
3. [Phase 2 -- Pipeline de preprocessing](#3-phase-2----pipeline-de-preprocessing)
4. [Phase 3 -- Entrainement et selection du modele](#4-phase-3----entrainement-et-selection-du-modele)
5. [Phase 4 -- Explainability (SHAP)](#5-phase-4----explainability-shap)
6. [Phase 5 -- Application web Flask](#6-phase-5----application-web-flask)
7. [Phase 6 -- Tests et CI/CD](#7-phase-6----tests-et-cicd)
8. [Phase 7 -- Revue finale et refactoring](#8-phase-7----revue-finale-et-refactoring)
9. [Bilan](#9-bilan)

---

## 1. Workflow general

```
  Developpeur                          Agent IA
  -----------                          --------
  Comprend la tache assignee
  Cree la structure de dossiers
  et les fichiers vides
          ---- prompt -------->
                                       Implemente les fonctions
                                       dans les fichiers existants
          <--- code genere ----
  Revue de code
  Validation / demande
  de corrections
          ---- correction ---->
                                       Corrige le code
          <--- code corrige ---
  Merge dans main
```

On repete ce cycle plusieurs fois par tache. Ca peut aller de 2-3
aller-retours pour un truc simple a une dizaine pour le frontend.

Les outils qu'on a utilises :

| Outil | Usage |
|:------|:------|
| Claude (Anthropic) | Agent principal, il travaille directement dans les fichiers du projet |
| ChatGPT (OpenAI) | Questions ponctuelles sur des API, brainstorming initial |
| GitHub Copilot | Autocompletion dans l'editeur pour les petites modifs |

Les sections suivantes montrent les prompts envoyes et ce qu'on a
obtenu. Les prompts sont paraphrases pour etre lisibles, mais le fond
est le meme.

---

## 2. Phase 1 -- Exploration des donnees (EDA)

Fichier concerne : `notebooks/eda.ipynb`

### Prompt 1 -- Structure initiale du notebook

> J'ai cree un notebook `notebooks/eda.ipynb` avec les sections suivantes
> en markdown : chargement, valeurs manquantes, equilibre des classes,
> outliers, correlations, optimisation memoire. Pour chaque section,
> implemente le code correspondant en utilisant le dataset UCI #938
> (Regensburg Pediatric Appendicitis). Utilise seaborn et matplotlib
> pour les visualisations. Sauvegarde les figures dans `reports/figures/`.

L'agent a produit tout le code d'un coup : telechargement via `ucimlrepo`,
heatmaps de valeurs manquantes, barplots de classes, boxplots, matrice de
Pearson, tests Mann-Whitney U. Il avait meme choisi une palette coherente
sans qu'on le demande (rouge = appendicite, vert = sain). On n'a rien
eu a retoucher sur ce premier jet.

### Prompt 2 -- Imputation par regression lineaire

> Dans la section "Valeurs manquantes", ajoute une demonstration de
> l'imputation par regression lineaire. Identifie les paires de variables
> avec une correlation >= 0.80, montre un scatter plot avec la droite de
> regression, et compare la distribution des valeurs predites vs originales.

Ca a marche du premier coup. Scatter plot, histogramme de densite, et
un fallback propre quand aucune paire ne contient de NaN.

### Prompt 3 -- Section equilibre des classes

> Pour la section "Equilibre des classes", analyse les 3 variables cibles
> (Diagnosis, Severity, Management). Pour chacune, calcule le ratio
> max/min des effectifs. Utilise un seuil de 1.5 pour determiner si
> c'est equilibre. Affiche les barplots avec les pourcentages et un
> resume des decisions prises.

L'agent a bien separe l'analyse par variable cible. Il a conclu que
Diagnosis (ratio 1.46) etait equilibre et que Severity et Management
ne l'etaient pas. On a ajoute nous-memes la note sur `class_weight='balanced'`
pour le SVM, parce que l'agent ne connaissait pas notre plan de modelisation.

### Prompt 4 -- Detection des outliers

> Pour la section outliers, visualise la distribution de chaque variable
> numerique (histogramme + KDE). Calcule le skewness. Classe les variables
> en deux groupes : normales (|skewness| < 0.5, methode Z-score) et
> asymetriques (|skewness| >= 0.5, methode IQR). Applique un capping
> (winsorization) et montre les boxplots avant/apres.

Le code etait correct mais l'agent avait mis toute la logique dans une
seule cellule de 80 lignes. On a du demander un deuxieme prompt pour
separer la visualisation de l'analyse du traitement.

> Separe le code de la section outliers en deux cellules : une pour la
> visualisation des distributions et le calcul du skewness, une autre
> pour le capping. Ajoute un rapport des outliers traites entre les deux.

Apres cette separation, on avait 3 cellules bien distinctes : visualisation,
rapport, traitement. Beaucoup plus lisible.

### Prompt 5 -- Nettoyage esthetique

> Le notebook est fonctionnel mais les cellules de code contiennent des
> blocs de commentaires trop lourds (# ===...===). Deplace les
> explications dans les cellules markdown et garde seulement des
> commentaires inline courts dans le code.

On a du faire ce prompt parce que l'agent avait mis des bandeaux de
10 lignes de commentaires au debut de chaque cellule de code. Le resultat
etait illisible. Apres correction, les explications sont dans le markdown
et le code est propre.

### Prompt 6 -- Mise en forme du markdown

> Les cellules markdown du notebook sont mal formatees. Certaines
> utilisent # au lieu de ### pour les sous-sections. Des textes sont
> en brut sans mise en forme. Ajoute une table des matieres cliquable
> dans l'intro, utilise des tableaux markdown la ou c'est pertinent,
> et des blockquotes pour les notes.

L'agent a reformate toutes les cellules markdown. Il a ajoute la table
des matieres, converti les listes en puces propres, et utilise des tableaux
pour les sections comme les tests statistiques et l'optimisation memoire.
On a du corriger le titre de la section 4.2 qu'il avait mis en `# 4.2`
(H1) au lieu de `### 4.2`.

---

## 3. Phase 2 -- Pipeline de preprocessing

Fichiers concernes : `src/data_processing.py`, `src/config.py`

### Prompt 1 -- Squelette du module

> J'ai cree `src/data_processing.py` avec les signatures suivantes :
> `load_data()`, `optimize_memory(df)`, `clean_data(df)`,
> `preprocess_data(df)`. Implemente chaque fonction. `load_data` doit
> essayer UCI d'abord, sinon lire le CSV local. `clean_data` doit
> supprimer les colonnes a fuite, reconstruire le BMI, imputer par
> regression les paires correlees, faire du feature engineering
> (WBC_CRP_Ratio), capper les outliers par IQR, et supprimer les
> features redondantes.

L'agent a implemente les 4 fonctions avec 9 etapes de nettoyage.
En revue de code on a change le seuil de correlation : l'IA l'avait mis
a 0.80, on l'a baisse a 0.69 pour capturer plus de paires. C'est le
genre de decision metier que l'IA ne peut pas prendre seule.

### Prompt 2 -- Optimisation memoire

> La fonction `optimize_memory` doit parcourir chaque colonne : convertir
> float64 en float32, int64 en int32, et les colonnes object avec moins
> de 50% de valeurs uniques en category. Affiche la memoire avant et
> apres.

L'agent a produit le code attendu. La reduction memoire est de l'ordre
de 50% sur notre dataset. On n'a rien change.

### Prompt 3 -- Centraliser la configuration

> Deplace tous les hyperparametres hardcodes (RANDOM_STATE, TEST_SIZE,
> seuils, chemins de fichiers) dans un fichier `src/config.py` separe.

Fait sans probleme. L'agent a cree le fichier et corrige les imports
dans les modules qui dependaient de ces constantes. On avait des valeurs
en dur un peu partout dans `data_processing.py`, ca les a toutes
centralisees.

### Prompt 4 -- Bug pandas 2.3

> J'ai une erreur sur `np.fill_diagonal` avec pandas 2.3 : le DataFrame
> retourne une vue read-only. Corrige ca.

L'agent a ajoute `.to_numpy()` avant l'appel a `np.fill_diagonal` pour
forcer une copie mutable. Le bug venait du Copy-on-Write introduit dans
pandas 2.3, que l'IA connaissait. Corrige du premier coup.

---

## 4. Phase 3 -- Entrainement et selection du modele

Fichiers concernes : `src/train_model.py`, `src/tuning.py`, `src/run.py`

### Prompt 1 -- Comparaison multi-modeles

> Implemente `src/train_model.py`. Le fichier doit entrainer 4 modeles
> (SVM RBF, Random Forest, LightGBM, CatBoost) avec cross-validation
> 5-fold sur le train set, calculer accuracy/precision/recall/F1/AUC
> sur le test set, puis selectionner le meilleur modele en priorisant
> le recall (contexte medical : il faut minimiser les faux negatifs).
> Sauvegarde le modele, le scaler, les feature names et les metriques
> dans `models/`.

L'agent a decompose le code en `get_models()`, `train_and_evaluate()`,
`select_best_model()` et `save_artifacts()`. La selection lexicographique
(recall > precision > AUC) etait correcte du premier coup.

En revue, on a ajoute `min_samples_split=5` au Random Forest. L'IA avait
laisse le defaut (2), ce qui overfittait sur nos 782 patients.

### Prompt 2 -- Seuil de classification optimal

> Apres le training, calcule le seuil optimal de classification en
> maximisant le F1-score sur les probabilites predites. Sauvegarde le
> seuil dans `models/threshold.txt`.

L'agent a utilise `precision_recall_curve` de scikit-learn, calcule le
F1 pour chaque seuil, et sauvegarde le meilleur (0.498). On a verifie
que le seuil etait coherent avec la distribution des probabilites.

### Prompt 3 -- Grid search

> Cree `src/tuning.py` qui fait un grid search cartesien pour les 4
> modeles avec stratified 5-fold CV. Definis des grilles raisonnables
> pour chaque modele. Sauvegarde les resultats dans
> `models/param_search_results.json` avec les top-3 par modele.

La structure etait bonne mais les grilles etaient trop etroites. On a
elargi le Random Forest (ajout `max_depth=None` et `n_estimators=300`)
et le CatBoost (ajout `iterations=400`). L'agent avait aussi oublie
`class_weight='balanced'` dans la grille SVM, on l'a ajoute.

### Prompt 4 -- Point d'entree principal

> Cree `src/run.py` qui orchestre tout : load_data, optimize_memory,
> clean_data, preprocess_data, puis appelle train_model.main(). Ce
> fichier doit etre le point d'entree principal.

Script court, rien a redire. L'agent a aussi ajoute un `__main__`
block dans `train_model.py` pour l'execution directe, ce qu'on
n'avait pas demande mais qui s'est avere utile.

### Prompt 5 -- Rapport final lisible

> A la fin du training, affiche un tableau comparatif des 4 modeles
> avec toutes les metriques, et marque le meilleur avec une etoile.

L'agent a produit un tableau formate avec `print()` et des colonnes
alignees. Il a aussi ajoute le `classification_report` de scikit-learn
pour le modele retenu. Correct du premier coup.

---

## 5. Phase 4 -- Explainability (SHAP)

Fichiers concernes : `src/evaluate_model.py`, `app/shap_utils.py`

### Prompt 1 -- Evaluation et plots SHAP globaux

> Dans `src/evaluate_model.py`, charge le modele sauvegarde, recalcule
> les metriques sur le test set, et genere les plots suivants dans
> `reports/images/` : matrice de confusion, courbe ROC, SHAP summary bar,
> SHAP beeswarm. Utilise TreeExplainer pour les modeles arborescents et
> KernelExplainer pour SVM.

L'agent a fait la detection automatique du type de modele pour choisir
le bon explainer. Fonctionnel du premier coup.

### Prompt 2 -- SHAP par prediction dans l'app

> Cree `app/shap_utils.py` avec deux fonctions : `init_explainer(model)`
> qui detecte le type de modele et initialise l'explainer SHAP, et
> `compute_shap_values(X_scaled, model, feature_names, explainer)` qui
> retourne les top-N features avec leur contribution. Traduis les noms
> de features en francais.

Ca a demande quelques aller-retours. Le premier code ne gerait pas le
format des SHAP values pour les classifieurs binaires (list vs array 3D),
et on a du ajouter la gestion du `VotingClassifier` nous-memes en revue.
Le dictionnaire de traduction FR etait correct.

### Prompt 3 -- Gestion des cas limites

> Le SHAP plante quand le modele est un VotingClassifier. Ajoute la
> detection de VotingClassifier dans init_explainer : extrais le
> premier sous-estimateur et utilise-le a la place.

L'agent a ajoute un `isinstance(model, VotingClassifier)` avec extraction
de `model.estimators_[0]`. On a verifie que les SHAP values restaient
coherentes apres cette extraction.

### Prompt 4 -- Waterfall pour un patient

> Ajoute un waterfall plot SHAP pour le patient de l'index 0 du test set.
> Sauvegarde dans `reports/images/shap_waterfall.png`.

L'agent a utilise `shap.plots.waterfall`. On a du ajouter
`matplotlib.pyplot.tight_layout()` parce que les labels etaient coupes.

---

## 6. Phase 5 -- Application web Flask

Fichiers concernes : `app/app.py`, `app/auth.py`, `app/config.py`,
templates, CSS, JS

### Prompt 1 -- Routes principales

> J'ai cree la structure suivante dans `app/` : `app.py`, `auth.py`,
> `config.py`, `templates/base.html`. Implemente `app.py` avec les
> routes `/` (landing), `/diagnosis` (formulaire), `/predict` (POST,
> prediction). Le formulaire doit avoir 3 etapes : donnees
> demographiques, symptomes cliniques, resultats de labo. La route
> predict doit construire le vecteur de features a partir du formulaire,
> appliquer le scaler, lancer la prediction, calculer les SHAP values,
> et afficher le resultat.

Le builder de feature vector etait le morceau le plus delicat. L'agent
devait gerer les features numeriques, les binaires (one-hot), et le ratio
WBC/CRP. En revue on a verifie que l'ordre des features correspondait
a `feature_names.pkl`, parce qu'un decalage d'une colonne fausse toute
la prediction. C'etait bon.

### Prompt 2 -- Configuration de l'app

> Cree `app/config.py` avec les chemins vers les artefacts du modele,
> la liste des features numeriques, un dictionnaire qui mappe les champs
> du formulaire aux noms de colonnes one-hot, et les parametres d'auth
> (longueur min username/password, identifiants admin par defaut).

L'agent a produit le fichier de config. On a ajoute WBC_CRP_SMOOTHING
(la constante 0.1 pour eviter la division par zero dans le ratio WBC/CRP)
qu'il avait laissee en dur dans `app.py`.

### Prompt 3 -- Authentification et historique

> Implemente `app/auth.py` comme un Blueprint Flask avec
> register/login/logout/profile/history/admin. Utilise Flask-Login
> et bcrypt pour le hash des mots de passe. Stocke les utilisateurs
> et l'historique des predictions dans SQLite. Cree un compte admin
> par defaut au premier demarrage.

L'agent a sorti le Blueprint complet, avec creation auto du compte
admin et historique filtrable. On a ajoute la validation de longueur
min pour username (3) et password (6) qu'il n'avait pas mise. On a
aussi rajoute la route `/profile` pour modifier ses identifiants, qu'il
avait oubliee malgre qu'elle etait dans le prompt.

### Prompt 4 -- Page de resultat avec SHAP

> La page de resultat doit afficher : la probabilite d'appendicite dans
> un anneau colore (rouge si haute, vert si basse), le niveau de risque
> (haut/modere/bas), et les SHAP values sous forme de barres horizontales
> animees. Pas d'image matplotlib pour le SHAP, tout en HTML/CSS.

C'est la ou l'agent s'est le plus plante. Le premier rendu affichait les
barres SHAP sans couleur (toutes grises). On a du preciser :

> Les barres SHAP doivent etre rouges pour les contributions positives
> (vers appendicite) et vertes pour les contributions negatives (vers
> sain). La largeur de chaque barre doit etre proportionnelle a la
> valeur absolue du SHAP value.

Apres cette correction, les barres etaient correctes. L'animation
d'entree (les barres qui s'etendent de 0 a leur valeur) marchait
du premier coup par contre.

### Prompt 5 -- Design frontend

> Cree les templates et les fichiers CSS/JS. Le design doit etre sombre
> avec un style glassmorphism. Le formulaire de diagnostic doit etre un
> wizard multi-etapes avec des boutons suivant/precedent. La landing page
> doit avoir un hero avec un slider.

C'est la partie ou on a le plus itere. Le premier rendu avait les bonnes
sections mais les couleurs ne collaient pas et les animations etaient
trop lentes. Ca a pris 4-5 passes pour arriver au resultat.

L'agent a produit 6 fichiers CSS modulaires (`core.css`, `form.css`,
`landing.css`, `pages.css`, `result.css`, `style.css`) et un JS par page
(`common.js`, `landing.js`, `diagnosis.js`, `result.js`, `history.js`,
`admin.js`).

### Prompt 6 -- Formulaire wizard

> Le formulaire de diagnostic doit calculer automatiquement l'age a
> partir de la date de naissance et le BMI a partir du poids et de la
> taille. Les validations doivent etre cote client. Les champs de
> l'etape 1 doivent etre valides avant de passer a l'etape 2.

L'agent a mis le calcul d'age et de BMI dans `diagnosis.js`. La
validation par etape marchait bien. On a du corriger un bug : il
calculait l'age en jours au lieu d'annees.

### Prompt 7 -- Traduction FR des features SHAP

> Dans la page resultat, les noms des features affichees doivent etre
> en francais. Cree un dictionnaire de traduction dans shap_utils.py.
> Par exemple : "WBC_Count" -> "Globules blancs (10^3/uL)",
> "Appendix_Diameter" -> "Diametre de l'appendice (mm)".

L'agent a cree le dictionnaire `FEATURE_NAMES_FR` avec les 21 features.
On a corrige quelques traductions approximatives (il avait traduit
"Neutrophil_Percentage" par "Pourcentage de neutrophiles" au lieu de
"Neutrophiles (%)").

---

## 7. Phase 6 -- Tests et CI/CD

Fichiers concernes : `tests/`, `.github/workflows/ci.yml`

### Prompt 1 -- Suite de tests

> Cree une suite de tests pytest couvrant tout le projet. Un fichier
> par module :
> - `test_data_processing.py` : chargement, nettoyage, preprocessing
> - `test_train_model.py` : entrainement, selection, artefacts
> - `test_evaluate_model.py` : metriques, generation de plots
> - `test_app.py` : routes Flask, builder de features, auth
> - `test_run.py` : integration du pipeline
> - `test_tuning.py` : grid search, scoring
>
> Utilise des fixtures session-scoped dans `conftest.py` pour ne pas
> recharger les donnees a chaque test.

80 tests generes d'un coup, repartis dans 6 fichiers. La plupart
passaient directement. On a rajoute manuellement des tests de validation
des artefacts (est-ce que les .pkl existent, est-ce que le nombre de
features correspond, est-ce que les metriques sont > 0.85).

### Prompt 2 -- Tests Flask specifiques

> Ajoute des tests pour le builder de feature vector dans app.py :
> verifie que les champs numeriques sont bien remplis, que le ratio
> WBC/CRP est calcule, que les features binaires sont correctement
> encodees en one-hot, et que les valeurs par defaut sont coherentes
> quand un champ est vide.

L'agent a ajoute une dizaine de tests dans `test_app.py`. Il a aussi
ajoute les tests pour le flow d'auth (register, login, acces a diagnosis
sans login, etc.).

### Prompt 3 -- Pipeline CI

> Cree `.github/workflows/ci.yml` avec des jobs separes pour :
> validation des imports, tests unitaires, pipeline de donnees,
> routes Flask, artefacts ML. Ajoute un job de release conditionnel
> sur les tags `v*`.

Workflow a 6 jobs avec cache pip. On a verifie que la chaine de
dependances etait correcte : release attend les 4 autres jobs. L'agent
utilisait `actions/checkout@v3` et `actions/setup-python@v4`, pas les
dernieres versions mais ca fonctionne.

### Prompt 4 -- Fix de tests qui echouent

> Les tests test_train_model.py::TestSavedModel echouent parce que
> le modele n'est pas retraine dans la CI. Fais en sorte que ces tests
> ne dependent que des artefacts deja commites dans models/.

L'agent a modifie les tests pour charger directement les .pkl au lieu
de relancer le pipeline. Ca a regle le probleme.

---

## 8. Phase 7 -- Revue finale et refactoring

### Prompt 1 -- Audit complet

> Parcours la totalite du projet (src/, app/, tests/, notebooks/,
> configurations). Releve tout ce qui pourrait poser probleme :
> imports manquants, incoherences entre modules, code mort,
> chemins hardcodes, failles de securite, fichiers oublies.
> Propose des corrections.

L'agent a trouve des trucs qu'on n'avait pas vus :

- imports redondants dans `train_model.py`
- `np.fill_diagonal` qui crashait sous pandas 2.3 avec Copy-on-Write
  (celui-ci avait deja ete corrige, mais il y avait un deuxieme
  endroit dans `evaluate_model.py`)
- des emojis dans les `print()` du notebook qui ne s'affichaient pas
  sur certains terminaux
- `favicon.svg` absent de la doc du README

### Prompt 2 -- Nettoyage du code mort

> Supprime tout le code mort, les imports inutilises, et les
> variables assignees mais jamais lues. Ne touche pas a la logique.

L'agent a nettoye une dizaine de lignes. Il a aussi supprime des
espaces en fin de ligne et des sauts de ligne excessifs. On a verifie
que les tests passaient toujours apres.

### Prompt 3 -- Coherence des versions

> Verifie que la version dans src/__init__.py correspond au badge du
> README et au tag git. Si ce n'est pas le cas, mets a jour.

L'agent a detecte un decalage entre `src/__init__.py` (1.2.4) et le
badge du README (1.1.2) et l'a corrige.

---

## 9. Bilan

### Ce qui a marche

On a vite compris qu'il fallait structurer avant de generer. Creer les
dossiers, les noms de fichiers et les signatures de fonctions en premier,
puis demander a l'IA de remplir. Sinon l'agent invente sa propre
architecture et ca part dans tous les sens.

Travailler avec des agents qui modifient directement les fichiers
(au lieu de copier-coller depuis un chat) a ete beaucoup plus rapide.
Moins d'erreurs de copie, moins de contexte perdu.

La revue de code, on l'a faite systematiquement. On a corrige des
seuils, ajoute des hyperparametres, ajuste des details metier. L'IA
ne connait pas nos contraintes cliniques.

### Ce qui a pose probleme

Sur les gros fichiers (300+ lignes), l'agent perd le fil. Il oublie
des imports, casse des dependances entre fonctions. La parade : decouper
en modifications ciblees, un bout a la fois.

L'agent a utilise des parametres SHAP qui n'existaient pas dans la
version qu'on avait d'installee. Les tests ont attrape ca, mais ca
aurait pu passer inapercu sans eux.

L'IA sur-ingenierie par defaut. Classes wrapper, decorateurs,
abstractions, alors qu'une fonction de 20 lignes suffit. Il faut lui
dire explicitement de faire simple.

Le frontend a demande le plus d'iterations. L'agent produit du HTML/CSS
structurellement correct mais esthetiquement plat. Les couleurs, les
espacements, les animations : tout ca demande plusieurs passes manuelles.

### Ce qu'on ferait differemment

- Ecrire les tests plus tot. On les a generes en phase 6, mais on
  aurait du les ecrire en meme temps que le code. Ca aurait evite
  des bugs qu'on a decouverts tard.
- Donner des exemples visuels (screenshots, mockups) a l'agent pour
  le frontend au lieu de decrire en texte. On a perdu du temps en
  aller-retours sur les couleurs et le layout.
- Preciser les versions des librairies dans les prompts. L'agent
  suppose des API recentes et ca peut casser avec des versions
  plus anciennes.
