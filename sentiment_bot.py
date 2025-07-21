import os
import re
import nltk
import joblib
from nltk.corpus import stopwords
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from pathlib import Path

print("Current directory:", Path.cwd())
print(".env file exists:", Path(".env").exists())

TOKEN = os.getenv("BOT_TOKEN")  # Make sure .env contains: BOT_TOKEN=your_token_here

print("Loaded token:", TOKEN)  # ✅ Now it's safe to print

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and tools
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hi! Send me a message and I'll analyze its sentiment.")

async def analyze_sentiment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    cleaned = clean_text(user_input)
    X = vectorizer.transform([cleaned])
    prediction = model.predict(X)[0]
    sentiment = label_encoder.inverse_transform([prediction])[0]
    await update.message.reply_text(f"🧠 Sentiment: *{sentiment.capitalize()}*", parse_mode='Markdown')

def main():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_sentiment))
    print("🤖 Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
