"""Interface utilisateur Gradio pour le pipeline RAG.

Le pipeline et ses modèles sont injectés depuis le notebook afin d'éviter de
charger une seconde fois le modèle Mistral sur le GPU Kaggle.

Gestion des trois états de réponse :
  - "answered"  : réponse normale avec sources.
  - "refused"   : question hors périmètre (détectée avant la recherche).
  - "abstained" : corpus insuffisant (détecté après reranking).
"""

from __future__ import annotations

import html
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Formatage des sources
# ---------------------------------------------------------------------------

def format_sources(result: Dict[str, Any]) -> str:
    """Formate le panneau 'Sources' selon le statut du résultat."""
    status = result.get("status", "answered")

    if status == "refused":
        return (
            "### ⛔ Question hors périmètre\n\n"
            "Le système n'a pas lancé de recherche pour cette question."
        )

    if status == "abstained":
        scope = result.get("scope_check", {})
        confidence = scope.get("confidence")
        score_text = (
            f"\n\nScore de pertinence du meilleur passage : `{float(confidence):.4f}` "
            f"(seuil : `{0.0}`)."
            if confidence is not None else ""
        )
        return (
            "### ⚠️ Documentation insuffisante\n\n"
            "Les passages récupérés n'étaient pas assez pertinents pour générer "
            f"une réponse fiable.{score_text}"
        )

    # Cas "answered" : affichage normal des sources
    sources = result.get("sources", [])
    scores = result.get("retrieval_scores", [])
    if not sources:
        return "Aucune source récupérée."

    lines = ["### Sources récupérées", ""]
    for rank, source in enumerate(sources, start=1):
        title = html.escape(str(source.get("doc_title") or "Document sans titre"))
        domain = html.escape(str(source.get("doc_source") or "Source non spécifiée"))
        score = scores[rank - 1] if rank - 1 < len(scores) else None
        score_text = f" — score {float(score):.4f}" if score is not None else ""
        lines.append(f"{rank}. **{title}**{score_text}  \n   `{domain}`")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Application Gradio
# ---------------------------------------------------------------------------

def create_gradio_app(pipeline: Any):
    """Construit l'application Gradio autour d'une instance RAG existante."""
    import gradio as gr

    def answer_question(question: str) -> Tuple[str, str]:
        question = (question or "").strip()
        if not question:
            return (
                "Veuillez saisir une question technique.",
                "",
            )
        try:
            result = pipeline.answer(question)
            status = result.get("status", "answered")
            answer_text = result.get("answer", "Aucune réponse generee.")

            # Préfixe visuel selon le statut
            if status == "refused":
                displayed_answer = f"⛔ {answer_text}"
            elif status == "abstained":
                displayed_answer = f"⚠️ {answer_text}"
            else:
                displayed_answer = answer_text

            return displayed_answer, format_sources(result)

        except Exception as exc:
            return (
                "Le systeme n'a pas pu traiter cette question. "
                "Consultez les logs pour le diagnostic.",
                f"Erreur : `{type(exc).__name__}: {exc}`",
            )

    with gr.Blocks(title="RAG — Documentation technique") as app:
        gr.Markdown(
            "# Assistant RAG — Documentation technique\n"
            "Posez une question sur **Python**, **Scikit-learn**, **LangChain** "
            "ou les **systèmes RAG**. "
            "La réponse est générée uniquement à partir de la documentation indexée."
        )
        question = gr.Textbox(
            label="Question",
            placeholder="Ex. Quelle est la difference entre fit() et fit_transform() ?",
            lines=3,
        )
        gr.Markdown("### Réponse")
        answer = gr.Markdown()
        gr.Markdown("### Sources")
        sources = gr.Markdown()
        with gr.Row():
            submit = gr.Button("Obtenir une réponse", variant="primary")
            clear = gr.ClearButton([question, answer, sources])
        submit.click(answer_question, inputs=question, outputs=[answer, sources])
        question.submit(answer_question, inputs=question, outputs=[answer, sources])
        gr.Markdown(
            "*Cette interface est une couche de démonstration. "
            "Les scores de retrieval ne constituent pas une probabilité de vérité.*"
        )
    return app


# ---------------------------------------------------------------------------
# Lancement
# ---------------------------------------------------------------------------

def launch_ui(
    pipeline: Any,
    share: bool = True,
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
):
    """Lance l'interface et retourne l'objet Gradio lancé."""
    app = create_gradio_app(pipeline)
    return app.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        show_error=True,
    )


if __name__ == '__main__':
    from etape5_generation import load_pipeline
    print('Chargement du pipeline...')
    pipeline = load_pipeline()
    print('Lancement de interface Gradio...')
    launch_ui(pipeline, share=False)
