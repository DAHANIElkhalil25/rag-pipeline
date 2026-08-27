# Rapport PFA RAG — compilation et personnalisation

Le fichier principal est `rapport_PFA_RAG.tex`. Il est autonome : la bibliographie IEEE est intégrée avec l’environnement `thebibliography`, donc aucun fichier `.bib` n’est requis.

## Compiler

Exécutez deux fois la commande suivante afin de mettre à jour la table des matières, les références et les listes :

```bash
pdflatex rapport_PFA_RAG.tex
pdflatex rapport_PFA_RAG.tex
```

La compilation peut également être effectuée dans Overleaf en important le dossier `rapport_pfa_latex` et en définissant `rapport_PFA_RAG.tex` comme document principal.

## Champs à vérifier avant dépôt

Les variables à personnaliser sont regroupées en haut du fichier : `\NomEtudiant`, `\Filiere`, `\AnneeUniversitaire`, `\EncadrantINSEA`, `\EncadrantOrganisme`, `\JureUn` et `\JureDeux`.

La page de garde est explicitement un **PFA**. Le jury contient exactement deux lignes INSEA : `M. Najib OURADI` et `M. Rachid BENMANSOUR`, comme demandé. Vérifiez leurs fonctions et l’orthographe institutionnelle avant dépôt officiel.

## Important sur les résultats

Le rapport décrit les résultats Ragas réellement reçus : les scores Ragas sont largement calculés, mais cinq lignes du jeu de test sont mal formées. Le texte les qualifie donc correctement de résultats **provisoires** et recommande de corriger les lignes avant de geler le score final. Cette transparence doit être conservée.

## Organisme et évaluation académique

Le contenu exploitable des deux supports transmis sur 3D Smart Factory a été reformulé dans le chapitre de présentation de l’organisme. Les captures d’écran ne sont pas intégrées au PDF, conformément à la consigne reçue.

Le fichier `REVUE_ACADEMIQUE_FINALE.md` contient l’évaluation indépendante du rapport, ses forces, les réserves méthodologiques et les vérifications restantes avant un dépôt officiel.
