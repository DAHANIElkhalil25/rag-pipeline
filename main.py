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
    python main.py --etape 6    # Étape 6 : Évaluation RAGAS du système

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


def check_dependencies():
    """Vérifie que toutes les dépendances sont installées."""
    missing = []
    for pkg, import_name in [
        ("gitpython", "git"),
        ("pandas", "pandas"),
        ("tqdm", "tqdm"),
        ("beautifulsoup4", "bs4"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"❌ Dépendances manquantes : {', '.join(missing)}")
        print(f"   Installez-les avec : pip install {' '.join(missing)}")
        sys.exit(1)

    # Optionnels
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


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline RAG — Collecte, préparation, benchmarking, "
                    "indexation, génération et évaluation"
    )
    parser.add_argument(
        "--etape", type=int, choices=[1, 2, 3, 4, 5, 6],
        help="Numéro de l'étape (1-6). Sans argument : toutes."
    )
    args = parser.parse_args()

    print_banner()
    print(f"⏰  Début : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("🔍  Vérification des dépendances...")
    check_dependencies()

    start = datetime.now()

    if args.etape == 1:
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
