import re
import nltk
import joblib
from nltk.corpus import stopwords
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Ensure stopwords are available
nltk.download('stopwords')

# Clean the incoming message
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

# Command handler for /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! Send me a review and I’ll tell you its sentiment.")

# Message handler
async def analyze_sentiment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    cleaned = clean_text(user_input)
    X = vectorizer.transform([cleaned])
    prediction = model.predict(X)[0]

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment = label_map.get(prediction, "Unknown")

    await update.message.reply_text(f"Sentiment: {sentiment}")

# Bot setup
def main():
    TOKEN = "8150065506:AAGO6vHdrv1Vegtb7k-fhT8u_vProZXRC1s"  # ← Replace this with your token
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_sentiment))

    print("🤖 Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
