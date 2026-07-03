import json
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

venv_site = root / "venv" / "lib"
if venv_site.exists():
    site_packages = sorted(venv_site.rglob("site-packages"))
    if site_packages:
        sys.path.insert(0, str(site_packages[0]))

import gradio as gr

from src.agent.retriever import CurriculumRetriever
from src.agent.orchestrator import TutorAgent
from src.ocr.ocr_pipeline import extract_text_from_image

retriever = CurriculumRetriever()
agent = TutorAgent(retriever)


class RateLimiter:
    def __init__(self, max_requests: int = 5, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window = window_seconds
        self.timestamps: list[float] = []

    def check(self) -> tuple[bool, str]:
        now = time.time()
        cutoff = now - self.window
        self.timestamps = [t for t in self.timestamps if t > cutoff]
        if len(self.timestamps) >= self.max_requests:
            retry_after = int(self.window - (now - self.timestamps[0]))
            return False, f"Rate limit reached ({self.max_requests} req/min). Please wait {retry_after} seconds."
        self.timestamps.append(now)
        return True, ""


limiter = RateLimiter()


def answer(question: str, language: str):
    allowed, msg = limiter.check()
    if not allowed:
        yield gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), msg
        return
    if not question.strip():
        yield gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), "Please enter a question first."
        return
    yield gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), "Thinking..."
    lang = "urdu" if language == "Urdu" else "english"
    result = agent.answer_question(question, lang)
    yield gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), result


def summarize(text: str):
    allowed, msg = limiter.check()
    if not allowed:
        yield gr.update(interactive=True), gr.update(interactive=True), msg
        return
    if not text.strip():
        yield gr.update(interactive=True), gr.update(interactive=True), "Please paste a lesson to summarize."
        return
    yield gr.update(interactive=False), gr.update(interactive=False), "Summarizing..."
    result = agent.summarize_lesson(text)
    yield gr.update(interactive=True), gr.update(interactive=True), result


def generate_quiz_data(topic: str, num: int):
    """Generate quiz and yield updates for all quiz UI components + quiz_state."""
    allowed, msg = limiter.check()
    hidden = [gr.update(visible=False)] * 14
    if not allowed:
        yield tuple([gr.update(interactive=True)] * 3 + [gr.update(value=msg, visible=True)] + hidden + [[]])
        return
    if not topic.strip():
        yield tuple([gr.update(interactive=True)] * 3 + [gr.update(value="Please enter a topic.", visible=True)] + hidden + [[]])
        return
    yield tuple([gr.update(interactive=False)] * 3 + [gr.update(value="Generating quiz...", visible=True)] + hidden + [[]])
    questions = agent.generate_quiz_json(topic, num)
    if not questions:
        yield tuple([gr.update(interactive=True)] * 3 + [gr.update(value="Could not generate quiz. Try a different topic.", visible=True)] + hidden + [[]])
        return
    component_updates = []
    for i in range(5):
        if i < len(questions):
            q = questions[i]
            lettered = [(f"{chr(65+j)}. {opt}", j) for j, opt in enumerate(q["options"])]
            component_updates.append(gr.update(value=f"**Q{i+1}:** {q['question']}", visible=True))
            component_updates.append(gr.update(choices=lettered, label="Select your answer", visible=True, value=None))
            component_updates.append(gr.update(value="", visible=False))
        else:
            component_updates += [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]
    yield tuple([gr.update(interactive=True)] * 3 + component_updates + [questions])


def check_quiz_answers(*args):
    """Grade the quiz — compare selected indices to correct indices."""
    *selections, quiz_state_raw = args
    if not quiz_state_raw:
        return "Generate a quiz first.", *([gr.update(visible=False)] * 5)
    questions = quiz_state_raw
    correct = 0
    total = len(questions)
    explain_updates = []
    for i in range(5):
        if i < total:
            q = questions[i]
            selected: int | None = selections[i]
            correct_idx = q["correct_index"]
            explanation = q.get("explanation", "")
            opts_html = ""
            for j, opt in enumerate(q["options"]):
                letter = chr(65 + j)
                if j == correct_idx and j == selected:
                    opts_html += f'<div style="color: #4ade80; padding: 2px 0;">{letter}. {opt} ✅ <span style="color: #94a3b8; font-size: 0.85em;">(correct)</span></div>'
                elif j == correct_idx:
                    opts_html += f'<div style="color: #4ade80; padding: 2px 0;">{letter}. {opt} ✅ <span style="color: #94a3b8; font-size: 0.85em;">(correct answer)</span></div>'
                elif j == selected:
                    opts_html += f'<div style="color: #f87171; padding: 2px 0;">{letter}. {opt} ❌</div>'
                else:
                    opts_html += f'<div style="color: #94a3b8; padding: 2px 0;">{letter}. {opt}</div>'
            if selected == correct_idx:
                correct += 1
            result_html = (
                f'<div style="background: #1e293b; padding: 10px; border-radius: 8px;">'
                f'{opts_html}'
                f'<div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #334155; color: #bfdbfe; font-size: 0.9em;">{explanation}</div>'
                f'</div>'
            )
            explain_updates.append(gr.update(value=result_html, visible=True))
        else:
            explain_updates.append(gr.update(visible=False))
    return f"### Score: {correct}/{total}", *explain_updates


def ocr(image_path: str):
    allowed, msg = limiter.check()
    if not allowed:
        yield gr.update(interactive=True), gr.update(interactive=True), msg
        return
    if not image_path:
        yield gr.update(interactive=True), gr.update(interactive=True), "Please upload an image first."
        return
    yield gr.update(interactive=False), gr.update(interactive=False), "Reading image..."
    result = extract_text_from_image(image_path)
    yield gr.update(interactive=True), gr.update(interactive=True), result


def send_to_ask(text: str):
    if not text.strip():
        return "", "Nothing to send — extract text first."
    return text, "Sent to Ask tab! Switch over and click Ask."


with gr.Blocks(title="TutorBuddy", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        "# TutorBuddy\n"
        "### Your friendly AI tutor for Grade 4–8 students\n"
        "Ask questions, summarize lessons, generate quizzes, and extract text from worksheets.\n"
        "Rate limit: 5 requests per minute."
    )

    with gr.Tab("Ask a Question"):
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="e.g. What is the water cycle?",
        )
        lang_dropdown = gr.Dropdown(
            ["English", "Urdu"], label="Language", value="English"
        )
        ask_btn = gr.Button("Ask", variant="primary")
        answer_output = gr.Textbox(label="Answer", lines=6)
        ask_btn.click(
            answer,
            inputs=[question_input, lang_dropdown],
            outputs=[question_input, lang_dropdown, ask_btn, answer_output],
        )

    with gr.Tab("Summarize Lesson"):
        lesson_input = gr.Textbox(
            label="Lesson Text", lines=8, placeholder="Paste the lesson here..."
        )
        summarize_btn = gr.Button("Summarize", variant="primary")
        summary_output = gr.Textbox(label="Summary", lines=6)
        summarize_btn.click(
            summarize,
            inputs=[lesson_input],
            outputs=[lesson_input, summarize_btn, summary_output],
        )

    with gr.Tab("Generate Quiz"):
        topic_input = gr.Textbox(
            label="Topic", placeholder="e.g. fractions, solar system"
        )
        num_slider = gr.Slider(1, 5, value=3, step=1, label="Number of Questions")
        quiz_btn = gr.Button("Generate Quiz", variant="primary")

        quiz_state = gr.State([])

        q1_md = gr.Markdown(visible=False)
        q1_radio = gr.Radio(choices=[], label="", visible=False)
        q1_explain = gr.Markdown(visible=False)

        q2_md = gr.Markdown(visible=False)
        q2_radio = gr.Radio(choices=[], label="", visible=False)
        q2_explain = gr.Markdown(visible=False)

        q3_md = gr.Markdown(visible=False)
        q3_radio = gr.Radio(choices=[], label="", visible=False)
        q3_explain = gr.Markdown(visible=False)

        q4_md = gr.Markdown(visible=False)
        q4_radio = gr.Radio(choices=[], label="", visible=False)
        q4_explain = gr.Markdown(visible=False)

        q5_md = gr.Markdown(visible=False)
        q5_radio = gr.Radio(choices=[], label="", visible=False)
        q5_explain = gr.Markdown(visible=False)

        quiz_btn.click(
            generate_quiz_data,
            inputs=[topic_input, num_slider],
            outputs=[
                topic_input, num_slider, quiz_btn,
                q1_md, q1_radio, q1_explain,
                q2_md, q2_radio, q2_explain,
                q3_md, q3_radio, q3_explain,
                q4_md, q4_radio, q4_explain,
                q5_md, q5_radio, q5_explain,
                quiz_state,
            ],
        )

        check_btn = gr.Button("Check Answers", variant="secondary")
        score_output = gr.Markdown("")
        check_btn.click(
            check_quiz_answers,
            inputs=[q1_radio, q2_radio, q3_radio, q4_radio, q5_radio, quiz_state],
            outputs=[score_output, q1_explain, q2_explain, q3_explain, q4_explain, q5_explain],
        )

    with gr.Tab("Worksheet OCR"):
        image_input = gr.Image(type="filepath", label="Upload Worksheet Image")
        ocr_btn = gr.Button("Extract Text", variant="primary")
        ocr_output = gr.Textbox(label="Extracted Text", lines=6)
        ocr_btn.click(
            ocr,
            inputs=[image_input],
            outputs=[image_input, ocr_btn, ocr_output],
        )
        send_btn = gr.Button("Send to Ask Tab")
        send_feedback = gr.Textbox(label="Status", lines=1, interactive=False)
        send_btn.click(
            send_to_ask,
            inputs=[ocr_output],
            outputs=[question_input, send_feedback],
        )

app.queue()
app.launch(share=False, server_port=7860)
