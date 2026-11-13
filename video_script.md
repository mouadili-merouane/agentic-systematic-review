# Script de la vidéo de démonstration (≤ 3 min)

**Préparation** : interface Gradio ouverte (dernière cellule du notebook → lien `*.gradio.live`), dépôt GitHub
ouvert dans un 2ᵉ onglet. Enregistrer avec OBS ou l'enregistreur Windows (Win + G), micro activé, en plein écran.
Répéter une fois à voix haute avec un chronomètre avant d'enregistrer. Les durées sont indicatives.

---

### 1 · Le problème — 0:00 → 0:20
**Écran** : onglet 1, schéma d'architecture.
> « Une revue systématique prend en moyenne plus d'un an : environ 67 semaines selon Borah et al., 2017. Nous avons
> construit un système multi-agents qui conduit une revue complète selon PRISMA 2020, de la question de recherche
> jusqu'au manuscrit, avec une validation humaine aux étapes critiques. »

### 2 · L'architecture — 0:20 → 0:45
**Écran** : rester sur le schéma, pointer les blocs avec la souris.
> « Le coordinateur LangGraph enchaîne des agents spécialisés : protocole, stratégie de recherche, recherche sur
> quatre bases, double screening avec un arbitre, puis un RAG hybride (FAISS, BM25, re-ranking) sur les textes
> intégraux, l'extraction avec évaluation du risque de biais, la synthèse, et un agent critique qui renvoie la
> synthèse en révision. Tout est journalisé et reprenable après une interruption. »

### 3 · Le pipeline en action — 0:45 → 1:30
**Écran** : cliquer sur **▶ Rejouer** (vitesse ≈ 10), laisser défiler 10 s ; puis onglet 2.
> « Voici la piste d'audit de l'exécution réelle : chaque agent, chaque décision, chaque validation humaine est
> horodatée. Ici, le protocole généré avec des critères numérotés, et les requêtes produites pour chaque base,
> par exemple la requête PubMed avec les termes MeSH. Nous avons identifié 285 enregistrements, 272 après
> dédoublonnage. »

### 4 · Transparence — 1:30 → 2:10
**Écran** : onglet 3, choisir un cas **arbitré** dans la liste déroulante ; puis onglet 4, choisir une étude.
> « Les deux screeners travaillent indépendamment : A est volontairement sensible, B strict. Leur accord est de
> κ = 0,84. Ici un désaccord : on voit le raisonnement de chacun et la décision motivée de l'arbitre. À l'extraction,
> chaque valeur est justifiée par une citation exacte de l'article, vérifiée automatiquement dans le texte source,
> avec l'évaluation PROBAST domaine par domaine. »

### 5 · Résultats — 2:10 → 2:40
**Écran** : onglet 5 (diagramme PRISMA + indicateurs), puis faire défiler l'onglet 6 (manuscrit).
> « Le diagramme PRISMA est généré automatiquement : 285 enregistrements, 71 études incluses. Pour rester dans le
> quota gratuit du modèle de langage, l'extraction et la synthèse portent sur un échantillon reproductible de
> 10 études ; un seul paramètre permet de toutes les traiter. 84 % des citations extraites sont retrouvées mot pour
> mot dans les articles, le manuscrit final ne contient aucune citation invalide et couvre les 16 éléments clés de
> PRISMA. L'agent critique a même détecté une AUC saisie en pourcentage, corrigée et tracée. Le tout en 2 heures de
> traitement actif contre environ 43 heures en manuel, soit environ 19 fois plus rapide. Sur 30 articles tirés au
> hasard et étiquetés en aveugle, le screening atteint 90 % d'exactitude, 100 % de spécificité et un kappa de 0,80. »

### 6 · Généralité et conclusion — 2:40 → 3:00
**Écran** : onglet 7, cliquer sur **Générer** avec une autre question ; finir sur la page GitHub.
> « Le système est générique : avec une nouvelle question, il génère le protocole et la stratégie de recherche en
> quelques secondes. Ses limites : seuls les textes en accès libre sont lus, et le manuscrit reste un brouillon à
> vérifier par un expert. Tout le code et les résultats sont sur GitHub. Merci. »

---
*Chiffres issus de `runs/demo_hf_ml/eval/evaluation_report.md`.*
