import gradio as gr
from transformers import BartTokenizerFast, BartForConditionalGeneration

# Path to your uploaded model folder (adjust if your folder name is different)
MODEL_DIR = "./bart_summarization_model_20250331_020400"

# Load tokenizer and model from local folder
tokenizer = BartTokenizerFast.from_pretrained(MODEL_DIR)
model = BartForConditionalGeneration.from_pretrained(MODEL_DIR)

# Define summarization function
def summarize(text):
    if not text.strip():
        return "⚠️ Please enter some text to summarize."
    
    # Tokenize and truncate to max 1024 tokens
    inputs = tokenizer([text], max_length=1024, truncation=True, return_tensors="pt")
    
    # Generate summary
    summary_ids = model.generate(
        **inputs,
        max_length=256,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    
    # Decode summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Gradio Interface
gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(lines=10, label="Paste News Article Here"),
    outputs=gr.Textbox(label="Generated Summary"),
    title="📰 NewsSense: BART Summarizer",
    description="Abstractive summarization of news articles using a fine-tuned BART model trained on Multi-News. Paste any long text or article to get a concise summary.",
    theme="default",
    allow_flagging="never"
).launch()
