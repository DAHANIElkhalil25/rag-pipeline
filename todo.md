# Finalisation sans nouvelle exécution Kaggle

- [x] Consolider les 90 décisions initiales et les 30 décisions de seconde passe.
- [x] Identifier précisément les questions encore dépourvues de preuve directe dans les candidats reçus.
- [x] Réviser seulement ces questions et leurs réponses de référence à partir de passages réels du corpus candidat.
- [x] Produire un fichier de revue de 120 enregistrements traçables, sans déclarer prématurément une validation humaine.
- [x] Ajouter les artefacts et contrôles au dépôt, puis expliquer la suite sans demander un nouveau lancement Kaggle.

# Validation experte demandée par l’étudiant

- [x] Définir et publier les critères de validation experte assistée par IA.
- [x] Contrôler les 120 questions, réponses, URLs et identifiants de chunks contre les candidats Kaggle reçus.
- [x] Geler le jeu final avec une provenance de validation explicite et non trompeuse.
- [x] Publier le jeu final et les contrôles, puis donner l’unique étape Kaggle pour Ragas.

# Correction de l’exécution Ragas finale

- [x] Diagnostiquer l’adaptateur d’embeddings incompatible avec Ragas 0.4.3 dans Kaggle.
- [x] Remplacer ou encapsuler l’adaptateur afin de fournir les méthodes asynchrones requises.
- [x] Ajouter un test de non-régression et mettre à jour le notebook français si nécessaire.
- [x] Publier la correction et donner une reprise limitée au mode EVALUATION_FINALE.

# Accompagnement pédagogique de l’exécution finale

- [ ] Expliquer en français simple chaque résultat ou erreur transmis depuis Kaggle.
- [ ] Interpréter l’archive Ragas finale et préparer les éléments utiles au rapport de stage.

# Reprise Kaggle à faible risque GPU

- [ ] Vérifier un chargement court de la correction Ragas sans reconstruire le pipeline ou l’index.
- [ ] Donner une seule instruction de reprise qui minimise l’attente et le risque d’expiration GPU.

# Correction Pydantic de l’adaptateur Ragas

- [x] Remplacer la sous-classe Pydantic incompatible par un adaptateur concret Ragas indépendant.
- [x] Tester réellement la construction de l’adaptateur et de AnswerRelevancy avant toute reprise Kaggle.
- [x] Publier la correction et ne donner une reprise qu’après ce test.

# Correction de la cellule de mise à jour Kaggle

- [x] Placer explicitement le processus dans `/kaggle/working` avant de remplacer le dossier cloné.
- [x] Rendre la mise à jour du dépôt idempotente et sans suppression du répertoire actif.
- [ ] Vérifier le notebook corrigé et ne donner qu’une reprise finale courte.

# Accord préalable sur les corrections

- [ ] Expliquer chaque changement de code ou de notebook, sa cause et son impact avant de le proposer.
- [ ] Obtenir l’accord explicite de l’étudiant avant toute nouvelle modification.

# Diagnostic du résultat Ragas incomplet

- [ ] Préserver et expliquer les métriques déterministes de récupération valides du run final.
- [ ] Lire les erreurs par échantillon Ragas avant de conclure sur les métriques génératives.
- [ ] Présenter une correction seulement après explication et accord explicite de l’étudiant.

# Correction autorisée du client Ragas

- [x] Remplacer uniquement le client OpenAI synchrone par `AsyncOpenAI` dans le juge Ragas.
- [x] Tester que les métriques Ragas reçoivent un client asynchrone compatible.
- [ ] Publier la correction et préparer une reprise finale sans préparation.
