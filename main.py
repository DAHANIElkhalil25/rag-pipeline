"""
main.py — Point d'entrée principal du pipeline RAG
====================================================
Exécute les étapes du pipeline séquentiellement ou individuellement.

Usage:
    python main.py              # Exécute toutes les étapes (1 à 6)
    python main.py --etape 1    # Étape 1 : Collecte des données
    python main.py --etape 2    # Étape 2 : Nettoyage et préparation
    python main.py --etape 3    # Étape 3 : Benchmarking des paramètres
    python main.py --etape 4    # Étape 4 : Chunking et indexation (config optimale)
    python main.py --etape 5    # Étape 5 : Recherche et génération RAG (LLM)
    python main.py --etape 6    # Baseline : évaluateur local Ragas-inspired
    python main.py --final-eval --dataset evaluation/datasets/dev_dataset_v1.jsonl \
        --run-id dev_v1 --judge-provider openai --judge-model gpt-4o-mini \
        --api-key-env OPENAI_API_KEY

Prérequis:
    pip install -r requirements.txt
    Pour l'étape 5-6 : GPU recommandé (Kaggle T4) + Mistral 7B
"""

import sys
import argparse
from datetime import datetime


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🤖  Pipeline RAG — Documentation Technique                ║
║                                                              ║
║   Système de Question-Réponse basé sur                       ║
║   Retrieval-Augmented Generation                             ║
║                                                              ║
║   Sources : Python · Scikit-learn · LangChain                ║
║                                                              ║
║   Réf. Lewis et al. (2020), Gao et al. (2024)               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


def check_dependencies(etape=None, final_eval=False):
    """Vérifie les dépendances nécessaires à l'étape demandée."""
    missing = []
    required = [
        ("gitpython", "git"),
        ("pandas", "pandas"),
        ("tqdm", "tqdm"),
        ("beautifulsoup4", "bs4"),
    ]
    if etape is None or etape >= 3:
        required.extend([
            ("numpy", "numpy"),
            ("tiktoken", "tiktoken"),
            ("sentence-transformers", "sentence_transformers"),
            ("faiss-cpu", "faiss"),
        ])
    if etape is None or etape >= 5:
        required.extend([
            ("torch", "torch"),
            ("transformers", "transformers"),
            ("accelerate", "accelerate"),
            ("requests", "requests"),
        ])
    if final_eval:
        required.extend([
            ("ragas", "ragas"),
            ("openai", "openai"),
        ])
    for pkg, import_name in required:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"❌ Dépendances manquantes : {', '.join(missing)}")
        print(f"   Installez-les avec : pip install {' '.join(missing)}")
        sys.exit(1)

    # Optionnel pour le nettoyage, mais recommandé.
    try:
        import ftfy
        print("  ✅ ftfy disponible (correction d'encodage avancée)")
    except ImportError:
        print("  ⚠️  ftfy non installé (optionnel) — pip install ftfy")

    print("  ✅ Toutes les dépendances requises sont installées.\n")


def run_etape1():
    """Exécute l'étape 1 : collecte des données."""
    from etape1_collecte import main as etape1_main
    etape1_main()


def run_etape2():
    """Exécute l'étape 2 : nettoyage et préparation."""
    from etape2_nettoyage import main as etape2_main
    etape2_main()


def run_etape3():
    """Exécute l'étape 3 : benchmarking des paramètres."""
    from etape3_benchmarking import main as etape3_main
    etape3_main()


def run_etape4():
    """Exécute l'étape 4 : chunking et indexation avec config optimale."""
    from etape4_indexation import main as etape4_main
    etape4_main()


def run_etape5():
    """Exécute l'étape 5 : recherche et génération RAG."""
    from etape5_generation import main as etape5_main
    etape5_main()


def run_etape6():
    """Exécute l'étape 6 : évaluation RAGAS du système."""
    from etape6_evaluation import main as etape6_main
    etape6_main()


def run_final_evaluation(dataset_path, run_id, provider, judge_model, api_key_env):
    """Lance l'évaluation finale officielle sans toucher à la baseline."""
    import asyncio
    import os
    from etape5_generation import load_pipeline
    from evaluation.ragas_runner import run_final_evaluation as run_ragas

    api_key = os.getenv(api_key_env, "")
    return asyncio.run(
        run_ragas(
            pipeline=load_pipeline(),
            dataset_path=dataset_path,
            run_id=run_id,
            provider=provider,
            judge_model=judge_model,
            api_key=api_key,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline RAG — Collecte, préparation, benchmarking, "
                    "indexation, génération et évaluation"
    )
    parser.add_argument(
        "--etape", type=int, choices=[1, 2, 3, 4, 5, 6],
        help="Numéro de l'étape (1-6). Sans argument : toutes."
    )
    parser.add_argument("--final-eval", action="store_true", help="Lance le protocole final Ragas officiel.")
    parser.add_argument("--dataset", type=str, help="Chemin JSONL du jeu dev ou test final.")
    parser.add_argument("--run-id", type=str, help="Identifiant unique de l'expérience finale.")
    parser.add_argument("--judge-provider", choices=("openai", "mistral"), help="Provider du modèle juge Ragas.")
    parser.add_argument("--judge-model", type=str, help="Nom du modèle juge.")
    parser.add_argument("--api-key-env", type=str, help="Nom de la variable d'environnement contenant la clé juge.")
    args = parser.parse_args()

    if args.final_eval and args.etape is not None:
        parser.error("Utilisez soit --etape, soit --final-eval, pas les deux.")
    if args.final_eval:
        missing_final_args = [
            name for name, value in {
                "--dataset": args.dataset,
                "--run-id": args.run_id,
                "--judge-provider": args.judge_provider,
                "--judge-model": args.judge_model,
                "--api-key-env": args.api_key_env,
            }.items() if not value
        ]
        if missing_final_args:
            parser.error(f"Arguments obligatoires pour --final-eval : {', '.join(missing_final_args)}")

    print_banner()
    print(f"⏰  Début : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("🔍  Vérification des dépendances...")
    check_dependencies(args.etape, final_eval=args.final_eval)

    start = datetime.now()

    if args.final_eval:
        from pathlib import Path
        final_result = run_final_evaluation(
            Path(args.dataset), args.run_id, args.judge_provider,
            args.judge_model, args.api_key_env,
        )
        print(f"✅ Évaluation finale sauvegardée : {final_result['run_dir']}")
    elif args.etape == 1:
        run_etape1()
    elif args.etape == 2:
        run_etape2()
    elif args.etape == 3:
        run_etape3()
    elif args.etape == 4:
        run_etape4()
    elif args.etape == 5:
        run_etape5()
    elif args.etape == 6:
        run_etape6()
    else:
        # Pipeline complet
        run_etape1()
        print("\n" + "━" * 65 + "\n")
        run_etape2()
        print("\n" + "━" * 65 + "\n")
        run_etape3()
        print("\n" + "━" * 65 + "\n")
        run_etape4()
        print("\n" + "━" * 65 + "\n")
        run_etape5()
        print("\n" + "━" * 65 + "\n")
        run_etape6()

    elapsed = datetime.now() - start
    minutes = elapsed.total_seconds() / 60

    print("━" * 65)
    print(f"⏱️   Temps total : {minutes:.1f} minutes")
    print(f"⏰  Fin : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("━" * 65)


if __name__ == "__main__":
    main()
