import streamlit as st
from transformers import pipeline
from collections import defaultdict
import re
import matplotlib.pyplot as plt
import numpy as np

# Load the pre-trained model and tokenizer (outside of any function)
emotion_pipe = pipeline("text-classification", model="monologg/bert-base-cased-goemotions-original", return_all_scores=True)

# Emotion mapping (define this outside the function)
emotion_map1 = {
    'joy': ['amusement', 'excitement'],
    'sadness': ['grief', 'disappointment'],
    'anger': ['annoyance', 'frustration'],
    'fear': ['nervousness', 'worry'],
    'surprise': ['realization'],
    'disgust': ['disgust', 'confusion'],
    'anticipation': ['desire', 'optimism'],
    'love': ['caring', 'affection'],
    'pride': ['pride'],
    'shame': ['embarrassment'],
    'guilt': ['remorse'],
    'relief': ['relief'],
    'gratitude': ['gratitude'],
    'hope': ['hope']
}

summarizero1 = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")

def preprocess_transcript(content):
    client_lines = []
    for line in content.split('\n'): # Split the content string into lines
      line = line.strip()
      if line.startswith("C:"):
        sentence = line[2:].strip()
        if sentence:
          client_lines.append(sentence)
    return client_lines

def classify_emotions(lines, max_length=500):
    emotion_over_time = {e: [] for e in emotion_map1.keys()}
    tokenizer = emotion_pipe.tokenizer
    for line in lines:
        tokens = tokenizer(line, truncation=True, max_length=max_length)
        truncated_line = tokenizer.decode(tokens["input_ids"])
        raw_results = emotion_pipe(truncated_line)[0]
        scores = defaultdict(float)
        for item in raw_results:
            for key, group in emotion_map1.items():
                if item['label'] in group:
                    scores[key] += item['score']
        total = sum(scores.values())
        if total > 0:
            for key in scores:
                scores[key] /= total
        for emotion in emotion_map1.keys():
            emotion_over_time[emotion].append(scores.get(emotion, 0.0))
    return emotion_over_time

def plot_emotions(emotion_scores):
    plt.figure(figsize=(12, 6))
    for emotion, scores in emotion_scores.items():
        plt.plot(scores, label=emotion.capitalize())
    plt.title("Client Emotion Trends Over Session")
    plt.xlabel("Client Utterance Index")
    plt.ylabel("Normalized Emotion Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)


def plot_emotions_spider(emotion_scores):
    labels = list(emotion_scores.keys())
    averages = [np.mean(emotion_scores[emotion]) for emotion in labels]
    values = averages + [averages[0]]
    labels = labels + [labels[0]]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=True)

    fig, ax = plt.subplots(figsize=(7, 12), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'b-', linewidth=2)
    ax.fill(angles, values, 'skyblue', alpha=0.4)
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels([label.capitalize() for label in labels])
    ax.set_title("Average Client Emotion Profile (Spider Graph)", size=9)
    plt.tight_layout()
    st.pyplot(plt)

def summarize_client_text(client_lines, chunk_size=500):
    text = " ".join(client_lines)
    summaries = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        result = summarizero1(chunk, max_length=130, min_length=30, do_sample=False)
        summaries.append(result[0]['summary_text'])
    return " ".join(summaries)

st.title("Emotion Analysis of Client Conversations")

uploaded_file = st.file_uploader("Choose a text file", type="txt")

if uploaded_file is not None:
    file_contents = uploaded_file.read().decode('utf-8')  # Decode the content
    client_lines = preprocess_transcript(file_contents)

    if client_lines:
        emotion_data = classify_emotions(client_lines)
        plot_emotions(emotion_data)
        plot_emotions_spider(emotion_data)

        summary = summarize_client_text(client_lines)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.write("No client lines found in the uploaded file. Please ensure the file follows the 'C:' format.")