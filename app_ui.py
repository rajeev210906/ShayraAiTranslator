#Completely Owned By Rajeev Ranjan No Copyrights Just Use it for any Purpose

from flask import Flask, request, render_template
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import langid

app = Flask(__name__)

# Load model and tokenizer
model = M2M100ForConditionalGeneration.from_pretrained("./DATA")
tokenizer = M2M100Tokenizer.from_pretrained("./DATA")

# Define language map
lang_map = {
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Arabic",
    "ast": "Asturian",
    "az": "Azerbaijani",
    "ba": "Bashkir",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan; Valencian",
    "ceb": "Cebuano",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "ff": "Fulah",
    "fi": "Finnish",
    "fr": "French",
    "fy": "Western Frisian",
    "ga": "Irish",
    "gd": "Gaelic; Scottish Gaelic",
    "gl": "Galician",
    "gu": "Gujarati",
    "ha": "Hausa",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "ht": "Haitian; Haitian Creole",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "ig": "Igbo",
    "ilo": "Iloko",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "jv": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Central Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "lb": "Luxembourgish; Letzeburgesch",
    "lg": "Ganda",
    "ln": "Lingala",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "my": "Burmese",
    "ne": "Nepali",
    "nl": "Dutch; Flemish",
    "no": "Norwegian",
    "ns": "Northern Sotho",
    "oc": "Occitan (post 1500)",
    "or": "Oriya",
    "pa": "Panjabi; Punjabi",
    "pl": "Polish",
    "ps": "Pushto; Pashto",
    "pt": "Portuguese",
    "ro": "Romanian; Moldavian; Moldovan",
    "ru": "Russian",
    "sd": "Sindhi",
    "si": "Sinhala; Sinhalese",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "ss": "Swati",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "th": "Thai",
    "tl": "Tagalog",
    "tn": "Tswana",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "wo": "Wolof",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "zh": "Chinese",
    "bn": "Bengali"
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_text = request.form["input_text"]
        target_lang_code = request.form["target_lang"]
        lang, _ = langid.classify(input_text)
        tokenizer.src_lang = lang
        tokenizer.tgt_lang = target_lang_code
        encoded_input = tokenizer(input_text, return_tensors="pt")
        generated_tokens = model.generate(**encoded_input, forced_bos_token_id=tokenizer.get_lang_id(target_lang_code))
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return render_template("index.html", translated_text=translated_text, target_lang=lang_map[target_lang_code], lang_map=lang_map)
    return render_template("index.html", lang_map=lang_map)

if __name__ == "__main__":
    app.run(debug=True)