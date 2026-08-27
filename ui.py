"""Interface utilisateur Gradio pour le pipeline RAG.

Le pipeline et ses modèles sont injectés depuis le notebook afin d'éviter de
charger une seconde fois le modèle Mistral sur le GPU Kaggle.
"""

from __future__ import annotations

import html
from typing import Any, Dict, List, Tuple


def format_sources(result: Dict[str, Any]) -> str:
    """Formate les sources récupérées en Markdown lisible."""
    if result.get("status") == "hors_perimetre":
        scope = result.get("scope_check", {})
        confidence = scope.get("confidence")
        confidence_text = (
            f"\n\nScore de récupération : `{float(confidence):.4f}`."
            if confidence is not None else ""
        )
        return (
            "### Statut : sujet hors périmètre ou source insuffisante\n\n"
            "Aucune source n’est affichée, car le système n’a pas généré de réponse factuelle."
            f"{confidence_text}"
        )
    sources = result.get("sources", [])
    scores = result.get("retrieval_scores", [])
    if not sources:
        return "Aucune source récupérée."

    lines = ["### Sources récupérées", ""]
    for rank, source in enumerate(sources, start=1):
        title = html.escape(str(source.get("doc_title") or "Document sans titre"))
        url = html.escape(str(source.get("doc_source") or "Source non spécifiée"))
        score = scores[rank - 1] if rank - 1 < len(scores) else None
        score_text = f" — score {float(score):.4f}" if score is not None else ""
        lines.append(f"{rank}. **{title}**{score_text}<br>\n   `{url}`")
    return "\n".join(lines)


def create_gradio_app(pipeline: Any):
    """Construit l'application Gradio autour d'une instance RAG existante."""
    import gradio as gr

    def answer_question(question: str) -> Tuple[str, str]:
        question = (question or "").strip()
        if not question:
            return "Veuillez saisir une question technique.", ""
        try:
            result = pipeline.answer(question)
            answer = result.get("answer", "Aucune réponse générée.")
            return answer, format_sources(result)
        except Exception as exc:
            return (
                "Le système n'a pas pu traiter cette question. Consultez les logs "
                "pour le diagnostic.",
                f"Erreur contrôlée : `{type(exc).__name__}: {exc}`",
            )

    with gr.Blocks(title="RAG — Documentation technique") as app:
        gr.Markdown(
            "# Assistant RAG — Documentation technique\n"
            "Posez une question sur **Python**, **Scikit-learn** ou **LangChain**. "
            "La réponse est générée uniquement à partir des passages récupérés."
        )
        question = gr.Textbox(
            label="Question",
            placeholder="Ex. Quelle est la différence entre fit() et fit_transform() ?",
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
            "*Cette interface est une couche de démonstration. Les scores de "
            "retrieval ne constituent pas une probabilité de vérité.*"
        )
    return app


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
