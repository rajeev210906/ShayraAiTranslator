from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import langid

# Specify the local path to the model and tokenizer
model_path = "."
tokenizer_path = "."

model = M2M100ForConditionalGeneration.from_pretrained(model_path)
tokenizer = M2M100Tokenizer.from_pretrained(tokenizer_path)

# Input
input_text = input("Enter text: ")

# Detect language of input text
lang, _ = langid.classify(input_text)

# Set source language
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

if lang in lang_map:
    print("Source language:", lang_map[lang])

# Ask user to select target language
print("Select target language:")
for i, (lang_code, lang_name) in enumerate(lang_map.items()):
    print(f"{i+1}. {lang_name} ({lang_code})")

target_lang_index = int(input("Enter the number of the target language: "))
target_lang_code = list(lang_map.keys())[target_lang_index - 1]

# Encode input text
tokenizer.src_lang = lang
tokenizer.tgt_lang = target_lang_code
encoded_input = tokenizer(input_text, return_tensors="pt")

# Generate translation
generated_tokens = model.generate(**encoded_input, forced_bos_token_id=tokenizer.get_lang_id(target_lang_code))
translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

print(f"Translated text ({lang_map[target_lang_code]}): {translated_text}")